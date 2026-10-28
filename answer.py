"""问答：检索 → 过滤掉明显不相关的 → 交给 DeepSeek 回答，并标出处。

设计要点（DESIGN.md 第 8 节）：
- 只根据检索到的资料回答；资料里没有，就老实说"资料里没有"，不编
- 回答里用 [1] [2] 这样的编号标出处，程序再把编号对应回"哪个文件、哪个标题"
- 相似度低于 MIN_SCORE 的直接丢掉；一块都不剩，就不调用 LLM（省钱，也避免它硬编）
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from search import run_search

# 📌 暂定：低于这个相似度的块直接丢掉。只看过一个样例（相关的 0.44~0.60，不相关的 0.34~0.38），
# 以后用评测集来定。可以用环境变量 RAG_MIN_SCORE 调整
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.35"))

NOT_FOUND = "根据现有资料无法回答这个问题。"

ANSWER_PROMPT = """你是一个笔记助手。请只根据下面【资料】中的内容回答【问题】。

要求：
1. 回答中用到哪条资料，就在那句话后面标上它的编号，例如 [1]、[2]。
2. 只用资料里有的信息，不要补充资料以外的知识，不要编造。
3. 如果资料里没有能回答问题的内容，只回答一句："{not_found}"
4. 用中文回答，简洁、准确。

【资料】
{sources}

【问题】
{question}"""


class AnswerError(RuntimeError):
    """没法回答：比如还没建库、embedding 模型对不上。"""


@dataclass
class Source:
    index: int              # 在提示词里的编号，对应回答里的 [1] [2]
    doc_name: str
    heading_path: str
    source_path: str
    score: float
    text: str
    cited: bool = False     # 回答里有没有真的引用它


@dataclass
class Answer:
    question: str
    answer: str
    found: bool                                   # 有没有找到相关资料
    sources: list[Source] = field(default_factory=list)


def build_prompt(question: str, sources: list[Source]) -> str:
    blocks = [f"[{s.index}] 来源：{s.heading_path}\n{s.text}" for s in sources]
    return ANSWER_PROMPT.format(not_found=NOT_FOUND, sources="\n\n".join(blocks), question=question)


def cited_indexes(answer: str) -> set[int]:
    """从回答里找出用到的编号，比如 "……[1][3]" → {1, 3}。"""
    return {int(n) for n in re.findall(r"\[(\d+)\]", answer)}


def answer_question(question: str, embedder, store, llm, k: int = 5,
                    min_score: float = MIN_SCORE) -> Answer:
    hits, error = run_search(question, embedder, store, k)
    if error:
        raise AnswerError(error)

    relevant = [h for h in hits if h.score >= min_score]
    if not relevant:
        return Answer(question=question, answer=NOT_FOUND, found=False)

    sources = [Source(index=i, doc_name=h.doc_name, heading_path=h.heading_path,
                      source_path=h.source_path, score=round(h.score, 4), text=h.text)
               for i, h in enumerate(relevant, 1)]
    text = llm.chat(build_prompt(question, sources), max_tokens=1500)

    used = cited_indexes(text)
    for s in sources:
        s.cited = s.index in used
    found = NOT_FOUND not in text
    return Answer(question=question, answer=text, found=found, sources=sources)
