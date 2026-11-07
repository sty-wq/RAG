"""本地 BM25 倒排索引；只索引 Milvus 已保存的文字，不修改数据库。

英文按单词/标识符切分，中文同时索引单字和相邻双字。不依赖分词模型，
分词规则固定，适合当前单机笔记库；这是 BM25 的词项定义，不是语义分词。
"""
from __future__ import annotations

from collections import Counter, defaultdict
import heapq
import math
import re
import unicodedata

TOKENIZER_VERSION = "cjk-unigram-bigram-identifiers-v1"
_TOKENS = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+|[a-z0-9_]+(?:[./-][a-z0-9_]+)*[+#]*")


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    tokens = []
    for match in _TOKENS.finditer(text):
        word = match.group()
        if "\u3400" <= word[0] <= "\ufaff":
            tokens.extend(word)
            tokens.extend(word[i:i + 2] for i in range(len(word) - 1))
        else:
            tokens.append(word)
    return tokens


class BM25Index:
    """BM25 Okapi，使用非负 IDF：log(1 + (N - df + .5) / (df + .5))。"""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 要求 k1 > 0、0 <= b <= 1，且参数有限")
        self.k1, self.b = k1, b
        self.size = len(texts)
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, text in enumerate(texts):
            terms = Counter(tokenize(text))
            self.lengths.append(terms.total())
            for term, frequency in terms.items():
                self.postings[term].append((index, frequency))
        self.average_length = sum(self.lengths) / max(self.size, 1) or 1.0

    def search(self, question: str, k: int = 5,
               allowed: set[int] | None = None) -> list[tuple[int, float]]:
        if k < 1:
            raise ValueError("k 必须大于 0")
        scores: dict[int, float] = defaultdict(float)
        # 查询里重复的词不重复计分；没有匹配词就返回空，不凑满 k 条零分结果。
        for term in sorted(set(tokenize(question))):
            postings = self.postings.get(term, ())
            df = len(postings)
            idf = math.log1p((self.size - df + 0.5) / (df + 0.5))
            for index, frequency in postings:
                if allowed is not None and index not in allowed:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
                scores[index] += idf * frequency * (self.k1 + 1) / (frequency + norm)
        return heapq.nsmallest(k, scores.items(), key=lambda item: (-item[1], item[0]))
