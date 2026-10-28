"""阶段四：入库 —— 把 chunk 和向量存进 Milvus，以及按向量检索。

默认用 Milvus Lite：它就是本地的一个 .db 文件，不用装服务器。接口（MilvusClient）和正式版
Milvus 一样，以后换成服务器版，只要把环境变量 MILVUS_URI 改成 http://地址:19530。

  MILVUS_URI   默认 <项目目录>/.rag_data/milvus.db

注意：同一个 .db 文件，同一时间只能被一个进程打开。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from pymilvus import DataType, MilvusClient

from chunks import Chunk

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_URI = str(PROJECT_DIR / ".rag_data" / "milvus.db")
COLLECTION = "notes"

# VARCHAR 字段的长度上限（按 UTF-8 字节算，一个汉字占 3 个字节）
MAX_BYTES = {"id": 64, "doc_id": 64, "doc_name": 512, "source_path": 1024,
             "heading_path": 2048, "text": 65535}
OUTPUT_FIELDS = ["doc_id", "doc_name", "source_path", "section_id", "heading_path", "chunk_order", "text"]


@dataclass
class SearchHit:
    score: float          # 余弦相似度，越接近 1 越相似
    chunk_id: str
    doc_name: str
    source_path: str
    heading_path: str
    chunk_order: int
    text: str


def fit_bytes(text: str, limit: int) -> tuple[str, bool]:
    """超过字节上限就截断（不会把一个汉字切成两半）。返回 (结果, 是否截断了)。"""
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text, False
    return data[:limit].decode("utf-8", errors="ignore"), True


class VectorStore:
    def __init__(self, uri: str | None = None, collection: str = COLLECTION):
        self.uri = uri or os.getenv("MILVUS_URI", DEFAULT_URI)
        self.collection = collection
        if self.uri.endswith(".db"):
            Path(self.uri).parent.mkdir(parents=True, exist_ok=True)   # Milvus Lite 要求目录已经存在
        self.client = MilvusClient(self.uri)

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

    def search(self, query_vector: list[float], k: int = 5) -> list[SearchHit]:
        """找出和问题向量最相似的 k 块。"""
        # Milvus 的数据平时存在磁盘上，检索前要先"加载"进内存。
        # 建库的那个进程里会自动加载，但换一个进程（比如单独运行 search.py）时，
        # 集合的状态是 released（在磁盘上、没加载），直接检索会报错。已经加载过的话，这一步什么也不做。
        self.client.load_collection(self.collection)
        results = self.client.search(self.collection, data=[query_vector], limit=k,
                                     output_fields=OUTPUT_FIELDS)
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
            ))
        return hits
