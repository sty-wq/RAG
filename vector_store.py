"""阶段四：入库 —— 把 chunk 和向量存进 Milvus，以及按向量检索。

默认用 Milvus Lite：它就是本地的一个 .db 文件，不用装服务器。接口（MilvusClient）和正式版
Milvus 一样，以后换成服务器版，只要把环境变量 RAG_MILVUS_URI 改成 http://地址:19530。

  RAG_MILVUS_URI   默认 <项目目录>/.rag_data/milvus.db

注意：同一个 .db 文件，同一时间只能被一个进程打开。
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

from chunks import Chunk

# PyMilvus 导入时会把 MILVUS_URI 当服务器地址解析，本地 .db 路径会使导入失败。
# 保留旧的本地变量写法，但在 SDK 导入前迁移到项目专用变量；新变量优先。
# 提前加载 SDK 本来也会加载的 .env，兼容其中的旧本地路径配置。
load_dotenv()
_legacy_uri = os.getenv("MILVUS_URI", "")
if _legacy_uri.endswith(".db") and "://" not in _legacy_uri:
    if not os.getenv("RAG_MILVUS_URI"):
        os.environ["RAG_MILVUS_URI"] = _legacy_uri
    # 不能 pop：SDK 的 load_dotenv() 会把 .env 里的旧值重新载入。
    os.environ["MILVUS_URI"] = ""

from pymilvus import DataType, MilvusClient  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_URI = str(PROJECT_DIR / ".rag_data" / "milvus.db")
COLLECTION = "notes"

# VARCHAR 字段的长度上限（按 UTF-8 字节算，一个汉字占 3 个字节）
MAX_BYTES = {"id": 64, "doc_id": 64, "doc_name": 512, "source_path": 1024,
             "heading_path": 2048, "text": 65535}
OUTPUT_FIELDS = ["doc_id", "doc_name", "source_path", "section_id", "heading_path", "chunk_order", "text"]


@dataclass
class SearchHit:
    score: float          # 最终排序分数；语义由 score_type 标明
    chunk_id: str
    doc_name: str
    source_path: str
    heading_path: str
    chunk_order: int
    text: str
    score_type: str = "cosine"
    dense_score: float | None = None
    lexical_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


def fit_bytes(text: str, limit: int) -> tuple[str, bool]:
    """超过字节上限就截断（不会把一个汉字切成两半）。返回 (结果, 是否截断了)。"""
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text, False
    return data[:limit].decode("utf-8", errors="ignore"), True


class VectorStore:
    def __init__(self, uri: str | None = None, collection: str = COLLECTION):
        self.uri = uri or os.getenv("RAG_MILVUS_URI") or os.getenv("MILVUS_URI") or DEFAULT_URI
        self.collection = collection
        if self.uri.endswith(".db"):
            Path(self.uri).parent.mkdir(parents=True, exist_ok=True)   # Milvus Lite 要求目录已经存在
        self.client = MilvusClient(self.uri)
        self._lexical_cache = None
        self._lexical_lock = threading.Lock()

    # 用 with 语句保证用完会关闭：Milvus Lite 同一时间只允许一个进程打开同一个文件
    def __enter__(self) -> VectorStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.client.close()

    @property
    def meta_path(self) -> Path:
        """索引的说明文件：记录用哪个模型、多少维建的库。"""
        if self.uri.endswith(".db"):
            return Path(self.uri + ".meta.json")                  # 和数据库文件放在一起
        return PROJECT_DIR / ".rag_data" / f"{self.collection}.meta.json"

    def read_meta(self) -> dict | None:
        if not self.meta_path.exists() or not self.client.has_collection(self.collection):
            return None
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def rebuild(self, chunks: list[Chunk], vectors: list[list[float]], embed_model: str) -> int:
        """全量重建（MVP 做法）：删掉旧集合，重新建表、插入。返回被截断的 chunk 数。"""
        if len(chunks) != len(vectors) or not chunks:
            raise ValueError("chunk 和向量的数量必须相同，且不能为空")
        dim = len(vectors[0])
        # 不能把上一次集合的 BM25 结果带到新集合。
        self._lexical_cache = None

        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=MAX_BYTES["id"])
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=MAX_BYTES["doc_id"])
        schema.add_field("doc_name", DataType.VARCHAR, max_length=MAX_BYTES["doc_name"])
        schema.add_field("source_path", DataType.VARCHAR, max_length=MAX_BYTES["source_path"])
        schema.add_field("section_id", DataType.INT64)
        schema.add_field("heading_path", DataType.VARCHAR, max_length=MAX_BYTES["heading_path"])
        schema.add_field("chunk_order", DataType.INT64)
        schema.add_field("text", DataType.VARCHAR, max_length=MAX_BYTES["text"])

        index_params = MilvusClient.prepare_index_params()
        # 余弦相似度：只看两个向量的方向是否接近，不看长度
        index_params.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
        self.client.create_collection(self.collection, schema=schema, index_params=index_params)

        rows, truncated = [], 0
        for chunk, vector in zip(chunks, vectors):
            row = {"vector": vector, "section_id": chunk.section_id, "chunk_order": chunk.order}
            cut_any = False
            for field in ("id", "doc_id", "doc_name", "source_path", "heading_path", "text"):
                row[field], cut = fit_bytes(getattr(chunk, field), MAX_BYTES[field])
                cut_any = cut_any or cut
            truncated += cut_any
            rows.append(row)

        for start in range(0, len(rows), 500):          # 分批插入，一次别太多
            self.client.insert(self.collection, rows[start:start + 500])
        self.client.flush(self.collection)              # 确保数据写进磁盘

        self.meta_path.write_text(json.dumps({
            "embed_model": embed_model,
            "dim": dim,
            "chunk_count": len(rows),
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return truncated

    def lexical_search(self, question: str, k: int = 5, doc_name: str | None = None) -> list[SearchHit]:
        """从已有集合构建进程内 BM25，首次使用时加载；无需重建向量或修改旧 schema。

        只读取 id、文字和出处，不读取向量。每个进程缓存一次；索引说明文件变化
        或本实例 rebuild() 后失效。当前集合应由本项目管理，不支持绕过它的外部写入。
        """
        from lexical import BM25Index

        stat = self.meta_path.stat() if self.meta_path.exists() else None
        stamp = (stat.st_mtime_ns, stat.st_size) if stat else None
        with self._lexical_lock:
            if self._lexical_cache is None or self._lexical_cache[0] != stamp:
                self.client.load_collection(self.collection)
                iterator = self.client.query_iterator(self.collection, batch_size=500,
                                                      output_fields=["id", *OUTPUT_FIELDS])
                hits = []
                try:
                    while batch := iterator.next():
                        for row in batch:
                            hits.append(SearchHit(score=0.0, chunk_id=row["id"], doc_name=row["doc_name"],
                                                  source_path=row["source_path"], heading_path=row["heading_path"],
                                                  chunk_order=row["chunk_order"], text=row["text"], score_type="bm25"))
                finally:
                    iterator.close()
                # Milvus 迭代顺序不保证稳定；固定次序后相同 BM25 分数才能稳定打破平局。
                hits.sort(key=lambda hit: hit.chunk_id)
                index = BM25Index([f"{hit.heading_path}\n{hit.text}" for hit in hits])
                documents: dict[str, set[int]] = {}
                for position, hit in enumerate(hits):
                    documents.setdefault(hit.doc_name, set()).add(position)
                self._lexical_cache = stamp, hits, index, documents
            _, hits, index, documents = self._lexical_cache
        allowed = documents.get(doc_name, set()) if doc_name is not None else None
        return [replace(hits[position], score=score, lexical_score=score)
                for position, score in index.search(question, k, allowed)]

    def search(self, query_vector: list[float], k: int = 5, doc_name: str | None = None) -> list[SearchHit]:
        """找出和问题向量最相似的 k 块。给了 doc_name，就只在这一篇文档里找。"""
        # Milvus 的数据平时存在磁盘上，检索前要先"加载"进内存。
        # 建库的那个进程里会自动加载，但换一个进程（比如单独运行 search.py）时，
        # 集合的状态是 released（在磁盘上、没加载），直接检索会报错。已经加载过的话，这一步什么也不做。
        self.client.load_collection(self.collection)
        # 过滤条件是 Milvus 的表达式，字符串要加双引号；json.dumps 会顺便把文件名里的引号转义掉
        flt = f"doc_name == {json.dumps(doc_name, ensure_ascii=False)}" if doc_name else ""
        results = self.client.search(self.collection, data=[query_vector], limit=k,
                                     filter=flt, output_fields=OUTPUT_FIELDS)
        hits = []
        for hit in results[0]:                          # results[i] 对应第 i 个问题
            entity = hit["entity"]
            hits.append(SearchHit(
                score=float(hit["distance"]),
                chunk_id=hit["id"],
                doc_name=entity["doc_name"],
                source_path=entity["source_path"],
                heading_path=entity["heading_path"],
                chunk_order=entity["chunk_order"],
                text=entity["text"],
                dense_score=float(hit["distance"]),
            ))
        return hits
