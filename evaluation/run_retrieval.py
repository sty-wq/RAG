"""检索评测：算 Hit@k、Recall@k、MRR，以及相似度门槛的效果。

两套评测集，用 --data 选：
  evaluation/data/crud    （默认）CRUD-RAG 新闻：按文档判断找没找对
  evaluation/data/qasper  Qasper 论文：按证据段落判断找没找对（题目里带 evidence 字段）

不调用 DeepSeek，结果每次都一样（只要语料、配置和模型不变）。

用法（在项目根目录运行；用本地 Ollama 时它要开着）：
  python -m evaluation.run_retrieval --label "v0.1.2 基线"
  python -m evaluation.run_retrieval --label "试试别的" --reuse-index   # 切分没变时，跳过重新建库
  python -m evaluation.run_retrieval --data evaluation/data/qasper --label "qasper 基线"

结果：
  evaluation/results/<时间>_<标签>.json   汇总分数（提交到 git，留作历史记录）
  <data>/runs/<时间>.jsonl                每道题的详细结果（不提交，用来分析错在哪）
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from statistics import median

from answer import MIN_SCORE
from chunks import MAX_CHUNK_CHARS
from evaluation.metrics import (average, covers, doc_covered, hit_at_k_covered, recall_at_k_covered,
                                reciprocal_rank_covered)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_DIR / "evaluation" / "data" / "crud"
RESULTS_DIR = PROJECT_DIR / "evaluation" / "results"
K_LIST = (1, 3, 5, 10)
# doc_hit：前 k 块里有没有来自"对的那篇文档"的块。按证据算分时它和 hit 不一样：
# doc_hit 高、hit 低，说明文档找对了、段落没找对，问题出在切块或排序上
METRICS = ([f"hit@{k}" for k in K_LIST] + [f"recall@{k}" for k in K_LIST] + ["mrr"]
           + ["doc_hit@1", "doc_hit@5"])


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def judge(q: dict, hits) -> tuple[list[set], list[set]] | None:
    """这道题的每个检索结果覆盖了哪些标准答案。返回 (covered, groups)；算不了分的题返回 None。

    - 题目带 evidence（Qasper）：标准答案是证据段落，第 j 段用编号 j 表示；一个标注者一组
    - 没有 evidence（CRUD-RAG）：标准答案是文档，只有一组，就是 gold_docs
    """
    if "evidence" in q:
        paragraphs = sorted({p for group in q["evidence"] for p in group})
        if not paragraphs:
            return None
        index = {p: j for j, p in enumerate(paragraphs)}
        groups = [{index[p] for p in group} for group in q["evidence"]]
        covered = [{j for j, p in enumerate(paragraphs) if covers(h.text, p)} for h in hits]
        return covered, groups
    if not q["gold_docs"]:
        return None
    gold = set(q["gold_docs"])
    return doc_covered([h.doc_name for h in hits], gold), [gold]


def should_answer(q: dict) -> bool:
    """CRUD-RAG 的题没有这个字段：没有标准答案文档的就是无答案题。"""
    return q.get("should_answer", bool(q["gold_docs"]))


def evaluate(questions: list[dict], embedder, store, k_max: int = 10,
             min_score: float = MIN_SCORE, within_doc: bool = False) -> tuple[dict, list[dict]]:
    """对每道题做一次检索、打分。embedder 和 store 从外面传进来，测试时可以换成假的。

    within_doc：只在这道题所属的文档里检索。Qasper 的题是拿着某一篇论文问的
    （"what was the size of the datasets?"），不说是哪篇，在整个库里根本没法找。
    """
    vectors = embedder.embed([q["question"] for q in questions])
    details = []
    for q, vector in zip(questions, vectors):
        if within_doc and q["gold_docs"]:
            hits = store.search(vector, k_max, doc_name=q["gold_docs"][0])
        else:
            hits = store.search(vector, k_max)
        ranked = [h.doc_name for h in hits]
        # group：分组报告分数用，默认就是题目类型。Qasper 补标题以后会分成 answerable:self / answerable:title
        row = {"qid": q["qid"], "type": q["type"], "group": q.get("group", q["type"]),
               "top1_score": round(hits[0].score, 4) if hits else 0.0,
               "ranked": ranked, "gold": q["gold_docs"], "should_answer": should_answer(q)}
        judged = judge(q, hits) if row["should_answer"] else None
        if judged is not None:
            covered, groups = judged
            row["relevant_ranks"] = [rank for rank, units in enumerate(covered, 1) if units]
            for k in K_LIST:
                row[f"hit@{k}"] = hit_at_k_covered(covered, k)
                row[f"recall@{k}"] = recall_at_k_covered(covered, groups, k)
            row["mrr"] = reciprocal_rank_covered(covered)
            gold_docs = set(q["gold_docs"])
            for k in (1, 5):
                row[f"doc_hit@{k}"] = 1.0 if gold_docs & set(ranked[:k]) else 0.0
        details.append(row)

    answerable = [r for r in details if "mrr" in r]                      # 能算检索分的题
    unanswerable = [r for r in details if not r["should_answer"]]        # 应该拒答的题
    groups = sorted(set(r["group"] for r in answerable))
    summary = {
        "counts": dict(Counter(r["group"] for r in details)),
        "overall": {"n": len(answerable), **average(answerable, METRICS)},
        "by_type": {g: {"n": sum(r["group"] == g for r in answerable),
                        **average([r for r in answerable if r["group"] == g], METRICS)} for g in groups},
        # 相似度门槛：top1 分数低于它，就直接回答"无法回答"、不调用 LLM
        "gate": {
            "min_score": min_score,
            "answerable_blocked": round(sum(r["top1_score"] < min_score for r in answerable)
                                        / max(len(answerable), 1), 4),      # 误拦：有答案却被拦下
            "unanswerable_blocked": round(sum(r["top1_score"] < min_score for r in unanswerable)
                                          / max(len(unanswerable), 1), 4),  # 拦对：该拒答的被拦下
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
    cols = ["hit@1", "hit@5", "recall@5", "recall@10", "mrr", "doc_hit@5"]
    print("| 类型 | 题数 | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    rows = [("全部", summary["overall"]), *summary["by_type"].items()]
    for name, m in rows:
        print(f"| {name} | {m['n']} | " + " | ".join(f"{m[c]:.3f}" for c in cols) + " |")
    print("\n各类题目数量：" + "，".join(f"{t} {n}" for t, n in summary["counts"].items()))
    g = summary["gate"]
    print(f"\n相似度门槛 {g['min_score']}：有答案的题被误拦 {g['answerable_blocked']:.1%}"
          f"（top1 中位数 {g['answerable_top1_median']}）；"
          f"该拒答的题被拦下 {g['unanswerable_blocked']:.1%}（{g['n_unanswerable']} 道，"
          f"top1 中位数 {g['unanswerable_top1_median']}）")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检索评测")
    parser.add_argument("--data", default=str(DEFAULT_DATA),
                        help="评测集目录：evaluation/data/crud（默认）或 evaluation/data/qasper")
    parser.add_argument("--label", required=True, help="这次评测的名字，比如 \"v0.1.2 基线\"")
    parser.add_argument("--questions", default="questions.jsonl",
                        help="用哪套题：questions.jsonl（易版，原题）或 questions_hard.jsonl（难版，口语化改写）")
    parser.add_argument("--reuse-index", action="store_true", help="已经建过库、切分方式也没变时，跳过重新建库")
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--min-score", type=float, default=MIN_SCORE,
                        help="相似度门槛。不同 embedding 模型的分数范围不一样，换模型后要重新定")
    parser.add_argument("--within-doc", action="store_true",
                        help="只在题目所属的文档里检索。⚠️ 假设已经知道是哪篇，真实使用时做不到，"
                             "只能当作理想情况下的上限参考")
    args = parser.parse_args(argv)

    from embedder import Embedder
    from ingest import run_ingest
    from vector_store import VectorStore

    data_dir = Path(args.data).resolve()
    questions = load_questions(data_dir / args.questions)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    embedder = Embedder()
    started = time.time()
    # 先记下 git 提交号再打开向量库：Milvus 连接开着的时候启动子进程，gRPC 会打印一堆 fork 相关的提示
    commit = git_commit()

    # 评测用单独的向量库文件，不会碰到你笔记的那个库
    with VectorStore(uri=str(data_dir / "index" / "milvus.db")) as store:
        if not args.reuse_index or store.read_meta() is None:
            print(f"正在建库：{manifest['corpus_docs']} 篇文档，embedding 模型 {embedder.model}，向量化可能要几分钟……")
            report = run_ingest(str(data_dir / "corpus"), embedder, store)
            print(f"建库完成：{report['chunks']} 块，用时 {time.time() - started:.0f} 秒")
        meta = store.read_meta()
        if meta["embed_model"] != embedder.model:
            print(f"向量库是用 {meta['embed_model']} 建的，现在配置的是 {embedder.model}，请去掉 --reuse-index 重新建库")
            return 1
        summary, details = evaluate(questions, embedder, store, args.k_max, args.min_score, args.within_doc)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    result = {
        "label": args.label,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": commit,
        "config": {"embed_model": embedder.model, "embed_base_url": embedder.base_url,
                   "max_chunk_chars": MAX_CHUNK_CHARS, "chunk_count": meta["chunk_count"], "k_max": args.k_max,
                   "within_doc": args.within_doc},
        "dataset": {"name": data_dir.name, "questions_file": args.questions,
                    **{k: manifest.get(k) for k in ("source", "source_sha256", "seed", "params", "questions",
                                                    "corpus_docs")}},
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

    print(f"\n【{args.label}】{result['git_commit']}，{meta['chunk_count']} 块，题目：{data_dir.name}/{args.questions}\n")
    print_table(summary)
    print(f"\n汇总已保存：{result_path.relative_to(PROJECT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
