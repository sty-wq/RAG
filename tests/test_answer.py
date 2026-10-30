"""问答逻辑：检索、过滤、提示词、出处。embedding、向量库、LLM 都是假的。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from answer import NOT_FOUND, AnswerError, Source, answer_question, build_prompt, cited_indexes


def hit(score: float, text: str, path: str = "笔记.md > 标题"):
    return SimpleNamespace(score=score, text=text, heading_path=path,
                           doc_name="笔记.md", source_path="notes/笔记.md")


class FakeEmbedder:
    model = "m"

    def embed_one(self, text):
        return [0.1, 0.2]


class FakeStore:
    def __init__(self, hits, meta=None):
        self.hits = hits
        self.meta = {"embed_model": "m"} if meta is None else meta

    def read_meta(self):
        return self.meta or None

    def search(self, vector, k):
        return self.hits[:k]


HITS = [hit(0.60, "野猪骑士用红黑树找下一跳", "题库.docx > 题目3"),
        hit(0.44, "代码解释：multiset 模拟红黑树", "题库.docx > 题目3"),
        hit(0.20, "完全不相关的内容", "其他.md")]


def test_cited_indexes():
    assert cited_indexes("用红黑树[1]，配合 multiset[3][1]。") == {1, 3}
    assert cited_indexes("没有引用") == set()


def test_prompt_numbers_sources_and_includes_question():
    sources = [Source(1, "a.md", "a.md > 甲", "notes/a.md", 0.6, "内容甲"),
               Source(2, "b.md", "b.md > 乙", "notes/b.md", 0.5, "内容乙")]
    prompt = build_prompt("问题是什么", sources)
    assert "[1] 来源：a.md > 甲\n内容甲" in prompt
    assert "[2] 来源：b.md > 乙\n内容乙" in prompt
    assert prompt.rstrip().endswith("问题是什么")
    assert NOT_FOUND in prompt                            # 告诉模型找不到时该怎么说
    assert "矛盾" in prompt                               # 资料自相矛盾时要指出来


def test_answer_with_citations_and_low_scores_dropped(fake_llm):
    llm = fake_llm("用红黑树找比当前值大的最小值[1]。")
    result = answer_question("红黑树怎么用", FakeEmbedder(), FakeStore(HITS), llm, k=5, min_score=0.35)

    assert result.found
    assert [s.text for s in result.sources] == [HITS[0].text, HITS[1].text]   # 0.20 那块被丢掉
    assert [s.cited for s in result.sources] == [True, False]
    assert "完全不相关" not in llm.prompts[0]


def test_nothing_relevant_means_no_llm_call(fake_llm):
    llm = fake_llm()                                       # 没准备回答：一旦被调用就会出错
    result = answer_question("天气怎么样", FakeEmbedder(), FakeStore([hit(0.1, "无关")]), llm)
    assert not result.found and result.answer == NOT_FOUND
    assert llm.prompts == []


def test_model_says_not_found(fake_llm):
    result = answer_question("问题", FakeEmbedder(), FakeStore(HITS), fake_llm(NOT_FOUND), min_score=0.35)
    assert not result.found
    assert not any(s.cited for s in result.sources)


def test_index_not_built_is_an_error(fake_llm):
    with pytest.raises(AnswerError, match="ingest"):
        answer_question("问题", FakeEmbedder(), FakeStore(HITS, meta={}), fake_llm())
