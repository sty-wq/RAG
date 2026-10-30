"""检索评测：在 CRUD-RAG 小样本上算 Hit@k、Recall@k、MRR，以及相似度门槛的效果。

不调用 DeepSeek，不花钱，结果每次都一样（只要语料、配置和模型不变）。

用法（在项目根目录运行，Ollama 要开着）：
  python -m evaluation.run_retrieval --label "v0.1.2 基线"
  python -m evaluation.run_retrieval --label "试试别的" --reuse-index   # 切分没变时，跳过重新建库

结果：
  evaluation/results/<时间>_<标签>.json   汇总分数（提交到 git，留作历史记录）
  evaluation/data/crud/runs/<时间>.jsonl  每道题的详细结果（不提交，用来分析错在哪）
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from statistics import median

from answer import MIN_SCORE
from chunks import MAX_CHUNK_CHARS
from evaluation.metrics import average, hit_at_k, recall_at_k, reciprocal_rank

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_DIR / "evaluation" / "data" / "crud"
RESULTS_DIR = PROJECT_DIR / "evaluation" / "results"
K_LIST = (1, 3, 5, 10)
METRICS = [f"hit@{k}" for k in K_LIST] + [f"recall@{k}" for k in K_LIST] + ["mrr"]
TYPES = ["1doc", "2docs", "3docs"]


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(questions: list[dict], embedder, store, k_max: int = 10,
             min_score: float = MIN_SCORE) -> tuple[dict, list[dict]]:
    """对每道题做一次检索、打分。embedder 和 store 从外面传进来，测试时可以换成假的。"""
    vectors = embedder.embed([q["question"] for q in questions])
    details = []
    for q, vector in zip(questions, vectors):
        hits = store.search(vector, k_max)
        ranked = [h.doc_name for h in hits]
        row = {"qid": q["qid"], "type": q["type"],
               "top1_score": round(hits[0].score, 4) if hits else 0.0,
               "ranked": ranked, "gold": q["gold_docs"]}
        if q["gold_docs"]:
            gold = set(q["gold_docs"])
            for k in K_LIST:
                row[f"hit@{k}"] = hit_at_k(ranked, gold, k)
                row[f"recall@{k}"] = recall_at_k(ranked, gold, k)
            row["mrr"] = reciprocal_rank(ranked, gold)
        details.append(row)

    answerable = [r for r in details if r["gold"]]
    unanswerable = [r for r in details if not r["gold"]]
    summary = {
        "overall": {"n": len(answerable), **average(answerable, METRICS)},
        "by_type": {t: {"n": sum(r["type"] == t for r in answerable),
                        **average([r for r in answerable if r["type"] == t], METRICS)} for t in TYPES},
        # 相似度门槛：top1 分数低于它，就直接回答"无法回答"、不调用 LLM
        "gate": {
            "min_score": min_score,
            "answerable_blocked": round(sum(r["top1_score"] < min_score for r in answerable)
                                        / max(len(answerable), 1), 4),      # 误拦：有答案却被拦下
            "unanswerable_blocked": round(sum(r["top1_score"] < min_score for r in unanswerable)
                                          / max(len(unanswerable), 1), 4),  # 拦对：没答案，被拦下
            "answerable_top1_median": round(median([r["top1_score"] for r in answerable]), 4) if answerable else None,
            "unanswerable_top1_median": round(median([r["top1_score"] for r in unanswerable]), 4) if unanswerable else None,
            "n_unanswerable": len(unanswerable),
        },
    }
    return summary, details


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_DIR,
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_DIR,
                               capture_output=True, text=True, check=True).stdout.strip()
        return out.stdout.strip() + ("（有未提交的改动）" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def print_table(summary: dict) -> None:
    cols = ["hit@1", "hit@5", "recall@5", "recall@10", "mrr"]
    print("| 类型 | 题数 | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    rows = [("全部", summary["overall"])] + [(t, summary["by_type"][t]) for t in TYPES]
    for name, m in rows:
        print(f"| {name} | {m['n']} | " + " | ".join(f"{m[c]:.3f}" for c in cols) + " |")
    g = summary["gate"]
    print(f"\n相似度门槛 {g['min_score']}：有答案的题被误拦 {g['answerable_blocked']:.1%}"
          f"（top1 中位数 {g['answerable_top1_median']}）；"
          f"无答案的题被拦下 {g['unanswerable_blocked']:.1%}（top1 中位数 {g['unanswerable_top1_median']}）")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检索评测")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="build_crud 生成的目录")
    parser.add_argument("--label", required=True, help="这次评测的名字，比如 \"v0.1.2 基线\"")
    parser.add_argument("--reuse-index", action="store_true", help="已经建过库、切分方式也没变时，跳过重新建库")
    parser.add_argument("--k-max", type=int, default=10)
    args = parser.parse_args(argv)

    from embedder import Embedder
    from ingest import run_ingest
    from vector_store import VectorStore

    data_dir = Path(args.data).resolve()
    questions = load_questions(data_dir / "questions.jsonl")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    embedder = Embedder()
    started = time.time()

    # 评测用单独的向量库文件，不会碰到你笔记的那个库
    with VectorStore(uri=str(data_dir / "index" / "milvus.db")) as store:
        if not args.reuse_index or store.read_meta() is None:
            print(f"正在建库：{manifest['corpus_docs']} 篇新闻，向量化可能要几分钟……")
            report = run_ingest(str(data_dir / "corpus"), embedder, store)
            print(f"建库完成：{report['chunks']} 块，用时 {time.time() - started:.0f} 秒")
        meta = store.read_meta()
        if meta["embed_model"] != embedder.model:
            print(f"向量库是用 {meta['embed_model']} 建的，现在配置的是 {embedder.model}，请去掉 --reuse-index 重新建库")
            return 1
        summary, details = evaluate(questions, embedder, store, args.k_max)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    result = {
        "label": args.label,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_commit(),
        "config": {"embed_model": embedder.model, "max_chunk_chars": MAX_CHUNK_CHARS,
                   "chunk_count": meta["chunk_count"], "k_max": args.k_max},
        "dataset": {k: manifest[k] for k in ("source", "source_sha256", "seed", "params", "questions", "corpus_docs")},
        **summary,
        "duration_s": round(time.time() - started, 1),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w.-]+", "_", args.label).strip("_")[:40]
    result_path = RESULTS_DIR / f"{stamp}_{slug}.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    runs_dir = data_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    with (runs_dir / f"{stamp}.jsonl").open("w", encoding="utf-8") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n【{args.label}】{result['git_commit']}，{meta['chunk_count']} 块\n")
    print_table(summary)
    print(f"\n汇总已保存：{result_path.relative_to(PROJECT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
