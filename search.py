"""检索小工具：dense / BM25 / 混合召回，以及可选的模型重排。

用法：
  python search.py "红黑树怎么找比当前值大的最小值"
  python search.py "chunk 太长会怎样" -k 3
  python search.py "requestId 怎么保证幂等" --mode hybrid --rerank local --candidate-k 50
"""
from __future__ import annotations

import argparse
import os

from retrieval import RetrievalConfig, add_retrieval_args, config_from_args, retrieve


def run_search(question: str, embedder, store, k: int = 5, *,
               config: RetrievalConfig | None = None, reranker=None):
    """返回 (检索结果, 错误说明)。出错时结果为空列表。"""
    config = config or RetrievalConfig()  # Python 调用保持原 dense 基线，CLI/API 显式传配置
    meta = store.read_meta()
    if meta is None:
        return [], "还没有建库，先运行：python ingest.py sample_notes"
    if config.mode != "bm25" and meta["embed_model"] != embedder.model:
        # 不同模型的向量不能放在一起比较，否则结果毫无意义
        return [], (f"向量库是用 {meta['embed_model']} 建的，现在配置的是 {embedder.model}。"
                    "请换回原来的模型，或者重新运行 ingest.py")
    vector = embedder.embed_one(question) if config.mode != "bm25" else None
    return retrieve(question, vector, store, k, config=config, reranker=reranker), None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="在向量库里检索最相关的几块")
    parser.add_argument("question", help="你的问题")
    parser.add_argument("-k", type=int, default=5, help="返回几块，默认 5")
    try:
        defaults = RetrievalConfig.from_env()
    except ValueError as exc:
        parser.error(str(exc))
    add_retrieval_args(parser, defaults=defaults, default_rerank=os.getenv("RERANK_BACKEND", "none"))
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        if not 1 <= args.k <= 1000:
            raise ValueError("k 必须在 1~1000 之间")
    except ValueError as exc:
        parser.error(str(exc))

    from embedder import Embedder
    from vector_store import VectorStore
    from reranker import create_reranker

    with VectorStore() as store:
        hits, error = run_search(args.question, Embedder(), store, args.k,
                                 config=config, reranker=create_reranker(args.rerank))
    if error:
        print(error)
        return 1

    print(f"问题：{args.question}\n")
    for rank, hit in enumerate(hits, 1):
        preview = hit.text[:80].replace("\n", " ⏎ ")
        print(f"[{rank}] {hit.score_type} {hit.score:.4f}　{hit.heading_path}")
        print(f"     {preview}…\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
