"""把评测题改写成"难版"：简短、口语化，像真实用户在搜索框里输入的问法。

为什么要做：CRUD-RAG 的题目是 GPT-4 照着新闻原文出的，又长又书面，日期、机构全称、
引号里的名称都和原文一模一样，检索起来太容易（基线 Hit@5 = 0.993，测不出优化效果）。

- 只改写"问题"，标准答案文档不变，所以不会引入标注错误
- 改写时 LLM **只看题目、不看新闻原文**，避免把原文里的信息带进新问题
- 结果有缓存（.rag_cache/question_rewrites.json）：重新抽样、或者再跑一次，都不会重复花钱
- 输出 questions_hard.jsonl（原题保留在 question_original 字段），
  以及 review_hard.md：一张"原题 / 改写 / 字面重合度"的对照表，方便人工抽查

用法（在项目根目录运行）：
  python -m evaluation.rewrite_questions
  python -m evaluation.run_retrieval --questions questions_hard.jsonl --label "v0.1.2 基线（难版）" --reuse-index
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

from evaluation.metrics import bigram_overlap
from llm import LLMError
from outline import JsonCache

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_DIR / "evaluation" / "data" / "crud"
CACHE_FILE = PROJECT_DIR / ".rag_cache" / "question_rewrites.json"
# 改了提示词就改这里，旧缓存自动失效
# v1 → v2：v1 允许"只保留最主要的一两个方面"，人工抽查发现 2docs / 3docs 的题会被删掉一问，
# 而被删掉的那一问往往正好对应其中一篇新闻 → 那篇新闻不再被题目需要，Recall 就被冤枉地拉低了。
# v2 要求每一问都保留。
PROMPT_VERSION = "v2"

REWRITE_PROMPT = """下面这道题是照着新闻原文出的，写得又长又书面。请把它改写成普通用户在搜索框里会输入的问法。

要求：
1. 简短、口语化，一般不超过 40 个字。
2. 问的还是同一件事，答案不能变。原题问到的每一个方面都要保留，一问都不能少，但可以压缩说法、合并成一句。
3. 不要照抄原题的长句子；去掉具体日期和机构全称，较长的名称改用简称或日常说法。
4. 不要添加原题里没有的信息。
5. 只输出改写后的问题本身，不要任何解释。

原题：{question}"""

_QUOTES = "\"'“”‘’「」『』"
_PREFIXES = ("改写后的问题：", "改写后：", "改写：", "问题：")


def clean(text: str) -> str:
    """只取第一行非空内容，去掉"改写："之类的前缀和两边的引号。"""
    line = next((x.strip() for x in text.splitlines() if x.strip()), "")
    for prefix in _PREFIXES:
        if line.startswith(prefix):
            line = line[len(prefix):].strip()
    return line.strip(_QUOTES).strip()


def cache_key(model: str, question: str) -> str:
    raw = f"{PROMPT_VERSION}|{model}|{question}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def rewrite_all(questions: list[dict], llm, cache: JsonCache, workers: int = 5) -> tuple[list[dict], dict]:
    """改写所有题目（有缓存的跳过）。返回 (难版题目, 统计)。"""
    todo = [q for q in questions if cache.get(cache_key(llm.model, q["question"])) is None]

    def call(q: dict):
        try:
            return q, clean(llm.chat(REWRITE_PROMPT.format(question=q["question"]), max_tokens=120)), None
        except LLMError as exc:
            return q, "", str(exc)

    # 多开几个线程同时请求，比一道一道等快得多；写缓存放回主线程，避免多个线程同时写文件
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(call, todo))
    errors = {}
    for q, text, error in results:
        if text:
            cache.put(cache_key(llm.model, q["question"]), text)
        else:
            errors[q["qid"]] = error or "改写结果为空"

    hard = []
    for q in questions:
        item = {**q, "question_original": q["question"]}
        new = cache.get(cache_key(llm.model, q["question"]))
        if new:
            item["question"] = new
        else:
            item["rewrite_failed"] = True          # 失败的保留原题，并做个标记
        hard.append(item)
    stats = {"total": len(hard), "new_calls": len(todo), "failed": len(errors), "errors": errors,
             "longer_than_original": sum(len(h["question"]) > len(h["question_original"]) for h in hard)}
    return hard, stats


def overlap_stats(questions: list[dict], corpus_dir: Path, field: str) -> float | None:
    """题目和它的标准答案文档之间，平均的字面重合度（无答案题没有文档，跳过）。"""
    values = []
    for q in questions:
        if q["gold_docs"]:
            docs = "\n".join((corpus_dir / name).read_text(encoding="utf-8") for name in q["gold_docs"])
            values.append(bigram_overlap(q[field], docs))
    return round(mean(values), 3) if values else None


def write_review(hard: list[dict], corpus_dir: Path, path: Path) -> None:
    lines = ["| qid | 原题 | 改写 | 重合度（原→改） |", "|---|---|---|---|"]
    for h in hard:
        if h["gold_docs"]:
            docs = "\n".join((corpus_dir / n).read_text(encoding="utf-8") for n in h["gold_docs"])
            score = f"{bigram_overlap(h['question_original'], docs):.2f} → {bigram_overlap(h['question'], docs):.2f}"
        else:
            score = "（无答案题）"
        cells = [h["qid"], h["question_original"], h["question"] + (" ⚠️改写失败" if h.get("rewrite_failed") else ""), score]
        lines.append("| " + " | ".join(c.replace("|", "｜").replace("\n", " ") for c in cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把评测题改写成简短、口语化的难版")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--workers", type=int, default=5, help="同时发几个请求")
    args = parser.parse_args(argv)

    from llm import DeepSeekClient

    data_dir = Path(args.data)
    questions = [json.loads(x) for x in (data_dir / "questions.jsonl").read_text(encoding="utf-8").splitlines() if x]
    llm = DeepSeekClient()
    print(f"改写 {len(questions)} 道题（模型：{llm.model}）……")
    hard, stats = rewrite_all(questions, llm, JsonCache(CACHE_FILE), args.workers)

    with (data_dir / "questions_hard.jsonl").open("w", encoding="utf-8") as f:
        for h in hard:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    write_review(hard, data_dir / "corpus", data_dir / "review_hard.md")

    corpus = data_dir / "corpus"
    print(f"\n完成：新调用 {stats['new_calls']} 次，失败 {stats['failed']} 道，比原题还长的 {stats['longer_than_original']} 道")
    for qid, error in stats["errors"].items():
        print(f"  ⚠️ {qid}：{error}")
    print(f"平均长度：原题 {mean(len(h['question_original']) for h in hard):.0f} 字 → "
          f"改写 {mean(len(h['question']) for h in hard):.0f} 字")
    print(f"和原文的字面重合度：原题 {overlap_stats(hard, corpus, 'question_original')} → "
          f"改写 {overlap_stats(hard, corpus, 'question')}")
    print("\n随机抽 8 道看看：")
    for h in random.Random(0).sample(hard, min(8, len(hard))):
        print(f"  [{h['qid']}]\n    原：{h['question_original']}\n    改：{h['question']}")
    print(f"\n完整对照表：{(data_dir / 'review_hard.md').relative_to(PROJECT_DIR)}")
    print(llm.usage_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
