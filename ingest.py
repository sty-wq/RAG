"""建库：解析 → 切章节 →（摘要 + 结构修正）→ 切 chunk → 向量化 → 存入 Milvus。

MVP 做法：每次全量重建。

用法：
  python ingest.py sample_notes            # 完整流程：摘要和结构修正会调用 DeepSeek（有缓存）
  python ingest.py sample_notes --no-llm   # 不调用 DeepSeek，只用规则切分，最快
"""
from __future__ import annotations

import argparse
import time

from chunker import split_sections
from chunks import Chunk, check_chunks_lossless, make_chunks
from main import load_directory


def run_ingest(directory: str, embedder, store, llm=None) -> dict:
    """完整的建库流程。embedder / store / llm 从外面传进来，测试时可以换成假的。"""
    documents, failed, empty = load_directory(directory)

    summary_cache = fix_cache = None
    if llm is not None:
        from outline import SUMMARY_CACHE_FILE, JsonCache, summarize_sections
        from structure_fix import FIX_CACHE_FILE, fix_structure
        summary_cache, fix_cache = JsonCache(SUMMARY_CACHE_FILE), JsonCache(FIX_CACHE_FILE)

    all_chunks: list[Chunk] = []
    fixes_applied = 0
    for doc in documents:
        sections = split_sections(doc)
        if llm is not None:
            summarize_sections(sections, llm, summary_cache)
            sections, fixes, _ = fix_structure(sections, llm, fix_cache)
            fixes_applied += len(fixes)
        chunks = make_chunks(doc, sections)
        if not check_chunks_lossless(sections, chunks):
            raise RuntimeError(f"{doc.name} 切 chunk 时丢了文字，停止建库")
        all_chunks.extend(chunks)

    report = {"documents": len(documents), "failed": failed, "empty": [d.name for d in empty],
              "chunks": len(all_chunks), "structure_fixes": fixes_applied, "truncated": 0}
    if not all_chunks:
        return report

    vectors = embedder.embed([c.embed_text for c in all_chunks])
    report["dim"] = len(vectors[0])
    report["truncated"] = store.rebuild(all_chunks, vectors, embedder.model)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把一个目录里的笔记建成向量库")
    parser.add_argument("directory", help="笔记所在的目录，例如 sample_notes")
    parser.add_argument("--no-llm", action="store_true", help="不调用 DeepSeek（不做摘要和结构修正）")
    args = parser.parse_args(argv)

    from embedder import Embedder
    from vector_store import VectorStore

    llm = None
    if not args.no_llm:
        from llm import DeepSeekClient
        llm = DeepSeekClient()

    embedder = Embedder()
    print(f"目录：{args.directory}　embedding 模型：{embedder.model}　"
          f"{'不调用 LLM' if llm is None else 'LLM：' + llm.model}")
    started = time.time()
    with VectorStore() as store:
        report = run_ingest(args.directory, embedder, store, llm)
        uri = store.uri

    print(f"\n文档 {report['documents']} 篇 → chunk {report['chunks']} 块"
          f"（向量 {report.get('dim', '-')} 维），用时 {time.time() - started:.1f} 秒")
    if llm is not None:
        print(f"结构修正 {report['structure_fixes']} 处；{llm.usage_report()}")
    for path, reason in report["failed"].items():
        print(f"  ⚠️ 失败：{path} —— {reason}")
    for name in report["empty"]:
        print(f"  ⚠️ 内容为空，已跳过：{name}")
    if report["truncated"]:
        print(f"  ⚠️ 有 {report['truncated']} 块因为太长被截断了")
    if report["chunks"] == 0:
        print("没有可以入库的内容。")
        return 1
    print(f"已写入：{uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
