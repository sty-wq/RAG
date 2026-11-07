"""检索评测的指标。都是纯函数，方便测试。

有两种"找对了"的判断方式：
- 按文档（CRUD-RAG）：这一块来自标准答案文档，就算找对了
- 按证据（Qasper）：这一块里包含标注好的证据段落，才算找对了。同一篇论文里别的段落不算

两种方式都先转成同一种形式再算分：
  covered：检索结果按顺序排列，第 i 个元素是"第 i 个 chunk 覆盖了哪些标准答案"（一个集合，可能是空的）
  groups： 标准答案分成几组。按文档时只有一组（这道题的标准答案文档）；
           按证据时一个标注者一组（Qasper 每道题有 2～3 个人分别标注证据，标得不完全一样）

注意 @k 里的 k 指的是"前 k 个 chunk"，因为交给 LLM 的就是前 k 个 chunk。
"""
from __future__ import annotations

import re
from statistics import mean

_NOT_WORD = re.compile(r"[\W_]+")      # 空白和标点（中文标点也算），汉字、字母、数字会保留
MIN_PIECE = 50                         # 一块是证据段落"切出来的一部分"时，至少要这么长才算（去掉空白后的字数）


def squash(text: str) -> str:
    """去掉所有空白：比较文字时不受空格、换行影响。"""
    return "".join(text.split())


def covers(chunk_text: str, paragraph: str, min_piece: int = MIN_PIECE) -> bool:
    """这一块是不是覆盖了这个证据段落。两种情况都算：
    1. 块里完整包含这个段落（短段落和别的段落合并成了一块）
    2. 这一块就是这个段落的一部分（长段落被按句子切成了好几块）。太短的碎片不算，免得偶然撞上
    """
    c, p = squash(chunk_text), squash(paragraph)
    if not c or not p:
        return False
    return p in c or (len(c) >= min(min_piece, len(p)) and c in p)


def hit_at_k_covered(covered: list[set], k: int) -> float:
    """前 k 个 chunk 里，有没有至少一个覆盖了标准答案。有就是 1，没有就是 0。"""
    return 1.0 if any(covered[:k]) else 0.0


def recall_at_k_covered(covered: list[set], groups: list[set], k: int) -> float:
    """前 k 个 chunk 覆盖了多少比例的标准答案。有好几组标注时，取最好的那一组。

    取"最好的一组"而不是"所有组的并集"：标注者 A 标了 2 段、B 标了 5 段，
    只要找全了 A 标的 2 段，就已经能回答问题了，不该因为 B 多标了几段而扣分。
    """
    groups = [g for g in groups if g]
    if not groups:
        return 0.0
    found = set().union(*covered[:k])
    return max(len(g & found) / len(g) for g in groups)


def reciprocal_rank_covered(covered: list[set]) -> float:
    """第一个"找对了"的结果排在第几位，取倒数：排第 1 得 1，排第 2 得 0.5，没找到得 0。"""
    for rank, units in enumerate(covered, 1):
        if units:
            return 1.0 / rank
    return 0.0


# ---- 按文档算分的简便写法（CRUD-RAG 用）----
# ranked：第 i 个 chunk 来自哪篇文档（同一篇可能出现好几次）；gold：标准答案文档

def doc_covered(ranked: list[str], gold: set[str]) -> list[set]:
    return [{doc} & gold for doc in ranked]


def hit_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return hit_at_k_covered(doc_covered(ranked, gold), k)


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """对需要综合 3 篇新闻才能回答的题，只找到 1 篇就是 1/3。"""
    return recall_at_k_covered(doc_covered(ranked, gold), [gold], k)


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    return reciprocal_rank_covered(doc_covered(ranked, gold))


def char_bigrams(text: str) -> set[str]:
    """去掉空白和标点后，所有相邻两个字组成的集合。中文没有空格分词，用"两个字一组"来比较最简单。"""
    flat = _NOT_WORD.sub("", text)
    return {flat[i:i + 2] for i in range(len(flat) - 1)}


def bigram_overlap(question: str, doc: str) -> float:
    """题目里的"两字组"，有多少比例原样出现在文档里。用来衡量题目和原文的字面重合度：
    越接近 1，越像是照着原文抄的；越低，越需要真正"理解意思"才能找到。"""
    q = char_bigrams(question)
    if not q:
        return 0.0
    return len(q & char_bigrams(doc)) / len(q)


def average(rows: list[dict], keys: list[str]) -> dict[str, float]:
    """把每道题的分数按指标取平均，保留 4 位小数。"""
    if not rows:
        return {key: 0.0 for key in keys}
    return {key: round(mean(row[key] for row in rows), 4) for key in keys}
