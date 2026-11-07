"""给 Qasper 的题目补上"是哪篇论文"，让它们能在整个库里检索。

问题：Qasper 的题是拿着某一篇论文问的（"what was the size of the datasets?"），
题目里不说是哪篇。把 416 篇放进一个库以后，这种题根本没法找。
而真实使用时，系统也不可能预先知道用户问的是哪篇。

做法：
1. 让 LLM 判断：只看这道问题、不知道是哪篇论文，能不能确定它问的是哪一篇？
   判断时 **LLM 只看问题，不看标题、正文和证据**
2. 不能确定的，用固定模板把论文标题写进问题：In the paper "<标题>", <原问题>
   用模板而不是让 LLM 改写：题目的意思一个字都不会变，结果也能复现
3. 能确定的，原样保留
4. 两类题分开报告分数（group 字段）：
   self   原题就能看出是哪篇。在整个库里找，别的论文也可能答得了，分数会偏低一些
   title  补了标题

结果有缓存（.rag_cache/qasper_context.json），重跑不重复花钱。
输出 questions_ctx.jsonl，以及 review_ctx.md（判断结果对照表，方便人工抽查）。

用法（在项目根目录运行）：
  python -m evaluation.contextualize_qasper
  python -m evaluation.run_retrieval --data evaluation/data/qasper --questions questions_ctx.jsonl \\
      --label "qasper 基线（补标题）" --reuse-index
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llm import LLMError
from outline import JsonCache

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_DIR / "evaluation" / "data" / "qasper"
CACHE_FILE = PROJECT_DIR / ".rag_cache" / "qasper_context.json"
PROMPT_VERSION = "v1"                  # 改了提示词就改这里，旧缓存自动失效

JUDGE_PROMPT = """下面是一道关于某篇英文论文的问题。资料库里有几百篇自然语言处理方向的论文。
请判断：只看这道问题本身、不知道是哪篇论文的情况下，能不能确定它问的是哪一篇？

判断标准：
- 问题里有代指（they、their、the authors、the dataset、the proposed model、this paper 等），指的是"某篇论文里的东西" → 不能确定
- 问的内容很多论文都会有（用了哪些数据集、比基线好多少、用了什么评价指标） → 不能确定
- 只有问题里出现了足以锁定这一篇论文的专有名称（特定的数据集名、模型名、任务名），才算能确定

只输出 JSON：{{"self_contained": true 或 false, "reason": "一句话理由"}}

问题：{question}"""


def cache_key(model: str, question: str) -> str:
    return hashlib.md5(f"{PROMPT_VERSION}|{model}|{question}".encode("utf-8")).hexdigest()


def paper_title(corpus_dir: Path, doc_name: str) -> str:
    """论文标题就是 Markdown 文件的第一行（build_qasper 写的 "# 标题"）。"""
    with (corpus_dir / doc_name).open(encoding="utf-8") as f:
        return f.readline().removeprefix("#").strip()


def with_title(question: str, title: str) -> str:
    """In the paper "<标题>", <原问题>。原问题的首字母改成小写，缩写（BERT、NLP）不动。"""
    q = question.strip()
    first = q.split(" ", 1)[0]
    if q[:1].isupper() and not (len(first) > 1 and first[1].isupper()):
        q = q[0].lower() + q[1:]
    return f'In the paper "{title}", {q}'


def judge_all(questions: list[dict], llm, cache: JsonCache, workers: int = 5) -> dict:
    """让 LLM 判断每道题能不能看出是哪篇（有缓存的跳过）。返回 {qid: 错误信息}。"""
    todo = [q for q in questions if cache.get(cache_key(llm.model, q["question"])) is None]

    def call(q: dict):
        try:
            data = llm.chat_json(JUDGE_PROMPT.format(question=q["question"]), max_tokens=200)
            if not isinstance(data.get("self_contained"), bool):
                return q, None, f"self_contained 不是 true/false：{data}"
            return q, {"self_contained": data["self_contained"], "reason": str(data.get("reason", ""))}, None
        except LLMError as exc:
            return q, None, str(exc)

    # 多开几个线程同时请求；写缓存放回主线程，避免多个线程同时写文件
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(call, todo))
    errors = {}
    for q, verdict, error in results:
        if verdict is not None:
            cache.put(cache_key(llm.model, q["question"]), verdict)
        else:
            errors[q["qid"]] = error
    return errors


def contextualize(questions: list[dict], corpus_dir: Path, llm, cache: JsonCache,
                  workers: int = 5) -> tuple[list[dict], dict]:
    errors = judge_all(questions, llm, cache, workers)
    out = []
    for q in questions:
        verdict = cache.get(cache_key(llm.model, q["question"]))
        item = {**q, "question_original": q["question"]}
        if verdict is None:
            # 判断失败：保守一点，当作看不出是哪篇，补上标题
            item["judge_failed"] = True
            self_contained, reason = False, ""
        else:
            self_contained, reason = verdict["self_contained"], verdict["reason"]
        item["self_contained"] = self_contained
        item["judge_reason"] = reason
        if not self_contained:
            item["question"] = with_title(q["question"], paper_title(corpus_dir, q["gold_docs"][0]))
        item["group"] = f"{q['type']}:{'self' if self_contained else 'title'}"
        out.append(item)
    stats = {"total": len(out), "failed": len(errors), "errors": errors,
             "groups": dict(Counter(item["group"] for item in out))}
    return out, stats


def write_review(items: list[dict], path: Path) -> None:
    lines = ["| qid | 能看出是哪篇？ | 理由 | 检索用的问题 |", "|---|---|---|---|"]
    for item in items:
        cells = [item["qid"], "✅ 能" if item["self_contained"] else "❌ 不能",
                 item["judge_reason"] + (" ⚠️判断失败" if item.get("judge_failed") else ""), item["question"]]
        lines.append("| " + " | ".join(c.replace("|", "｜").replace("\n", " ") for c in cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="给 Qasper 的题目补上是哪篇论文")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--workers", type=int, default=5, help="同时发几个请求")
    args = parser.parse_args(argv)

    from llm import DeepSeekClient

    data_dir = Path(args.data).resolve()
    questions = [json.loads(x) for x in (data_dir / "questions.jsonl").read_text(encoding="utf-8").splitlines() if x]
    llm = DeepSeekClient()
    print(f"判断 {len(questions)} 道题（模型：{llm.model}）……")
    items, stats = contextualize(questions, data_dir / "corpus", llm, JsonCache(CACHE_FILE), args.workers)

    with (data_dir / "questions_ctx.jsonl").open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    write_review(items, data_dir / "review_ctx.md")

    print(f"\n完成：失败 {stats['failed']} 道（按看不出是哪篇处理）")
    for qid, error in stats["errors"].items():
        print(f"  ⚠️ {qid}：{error}")
    print("各组题数：" + "，".join(f"{g} {n}" for g, n in sorted(stats["groups"].items())))
    print("\n随机抽 8 道看看：")
    for item in random.Random(0).sample(items, min(8, len(items))):
        mark = "能" if item["self_contained"] else "不能"
        print(f"  [{item['qid']}] {mark}：{item['judge_reason']}\n    {item['question']}")
    print(f"\n完整对照表：{(data_dir / 'review_ctx.md').relative_to(PROJECT_DIR)}")
    print(llm.usage_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
