"""检索评测的指标。都是纯函数，方便测试。

ranked：检索结果按顺序排列的"文档名"列表，第 i 个元素就是第 i 个 chunk 来自哪篇文档。
        同一篇文档可能出现好几次（它被切成了好几块）。
gold：  这道题的标准答案：应该被找到的那几篇文档。

注意 @k 里的 k 指的是"前 k 个 chunk"，因为交给 LLM 的就是前 k 个 chunk。
"""
from __future__ import annotations

from statistics import mean


def hit_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """前 k 个 chunk 里，有没有至少一篇标准答案文档。有就是 1，没有就是 0。"""
    return 1.0 if gold & set(ranked[:k]) else 0.0


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """前 k 个 chunk 覆盖了多少比例的标准答案文档。

    对需要综合 3 篇新闻才能回答的题，只找到 1 篇就是 1/3。
    """
    if not gold:
        return 0.0
    return len(gold & set(ranked[:k])) / len(gold)


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    """第一个"找对了"的结果排在第几位，取倒数：排第 1 得 1，排第 2 得 0.5，没找到得 0。

    对所有题目取平均，就是 MRR（Mean Reciprocal Rank）。
    """
    for rank, doc in enumerate(ranked, 1):
        if doc in gold:
            return 1.0 / rank
    return 0.0


def average(rows: list[dict], keys: list[str]) -> dict[str, float]:
    """把每道题的分数按指标取平均，保留 4 位小数。"""
    if not rows:
        return {key: 0.0 for key in keys}
    return {key: round(mean(row[key] for row in rows), 4) for key in keys}
