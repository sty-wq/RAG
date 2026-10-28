"""检索小工具：输入一个问题，看看向量库找出来的是哪几块。

还不接 DeepSeek 生成回答（那是 M3），先亲眼确认"找得准不准"。

用法：
  python search.py "红黑树怎么找比当前值大的最小值"
  python search.py "chunk 太长会怎样" -k 3
"""
from __future__ import annotations

import argparse


def run_search(question: str, embedder, store, k: int = 5):
    """返回 (检索结果, 错误说明)。出错时结果为空列表。"""
    meta = store.read_meta()
    if meta is None:
        return [], "还没有建库，先运行：python ingest.py sample_notes"
    if meta["embed_model"] != embedder.model:
        # 不同模型的向量不能放在一起比较，否则结果毫无意义
        return [], (f"向量库是用 {meta['embed_model']} 建的，现在配置的是 {embedder.model}。"
                    "请换回原来的模型，或者重新运行 ingest.py")
    return store.search(embedder.embed_one(question), k), None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="在向量库里检索最相关的几块")
    parser.add_argument("question", help="你的问题")
    parser.add_argument("-k", type=int, default=5, help="返回几块，默认 5")
    args = parser.parse_args(argv)

    from embedder import Embedder
    from vector_store import VectorStore

    with VectorStore() as store:
        hits, error = run_search(args.question, Embedder(), store, args.k)
    if error:
        print(error)
        return 1

    print(f"问题：{args.question}\n")
    for rank, hit in enumerate(hits, 1):
        preview = hit.text[:80].replace("\n", " ⏎ ")
        print(f"[{rank}] 相似度 {hit.score:.3f}　{hit.heading_path}")
        print(f"     {preview}…\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
