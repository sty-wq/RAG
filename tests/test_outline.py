"""阶段二 第 ② 步：写摘要、生成大纲。"""
from __future__ import annotations

from chunker import split_sections
from llm import LLMError
from outline import SHORT_CONTENT, JsonCache, build_outline, summarize_sections

LONG = "很长的内容。" * 40          # 超过 SHORT_CONTENT，需要总结


def test_only_long_sections_are_summarized(make_doc, fake_llm, tmp_path):
    doc = make_doc(f"# 父\n## 长\n{LONG}\n## 短\n一句话")
    sections = split_sections(doc)
    llm = fake_llm("长的摘要")

    stats = summarize_sections(sections, llm, JsonCache(tmp_path / "c.json"))

    assert len(LONG) >= SHORT_CONTENT
    assert stats == {"空": 2, "太短": 1, "缓存命中": 0, "新调用": 1, "失败": 0}   # 根节点和"父"都是空的
    assert sections[2].summary == "长的摘要"
    assert sections[3].summary is None


def test_summaries_are_cached(make_doc, fake_llm, tmp_path):
    cache = JsonCache(tmp_path / "c.json")
    summarize_sections(split_sections(make_doc(f"# A\n{LONG}")), fake_llm("摘要"), cache)

    again = split_sections(make_doc(f"# A\n{LONG}"))
    llm = fake_llm()                                         # 没有准备任何回答：一旦被调用就会报错
    stats = summarize_sections(again, llm, JsonCache(tmp_path / "c.json"))

    assert stats["缓存命中"] == 1 and llm.prompts == []
    assert again[1].summary == "摘要"


def test_one_failure_does_not_stop_the_others(make_doc, fake_llm, tmp_path):
    sections = split_sections(make_doc(f"# A\n{LONG}\n# B\n{LONG}x"))
    llm = fake_llm(LLMError("网络错误"), "B 的摘要")

    stats = summarize_sections(sections, llm, JsonCache(tmp_path / "c.json"))

    assert stats["失败"] == 1 and stats["新调用"] == 1
    assert sections[1].summary is None and sections[2].summary == "B 的摘要"


def test_outline_shows_summary_text_or_nothing(make_doc):
    sections = split_sections(make_doc(f"# 父\n## 长\n{LONG}\n## 短\n一句话"))
    sections[2].summary = "长的摘要"
    outline = build_outline(sections)
    assert "[1] L1 父 | 0 字 | （无内容）" in outline
    assert "摘要：长的摘要" in outline
    assert "原文：一句话" in outline
