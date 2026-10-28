"""阶段二 第 ②' 步：LLM 修正标题层级。"""
from __future__ import annotations

from chunker import check_lossless, split_sections
from llm import LLMError
from outline import JsonCache
from structure_fix import SectionFix, apply_fixes, fix_structure, validate_fixes

DOC = "# 题目1\n内容\n# 题目2\n内容\n# 参考代码\n代码"


def test_invalid_fixes_are_dropped_with_reasons(make_doc):
    sections = split_sections(make_doc(DOC))
    data = {"fixes": [
        {"section_id": 0, "new_level": 2},                  # 根节点不能改
        {"section_id": 9, "new_level": 2},                  # 不存在
        {"section_id": "3", "new_level": "2", "reason": "数字写成了字符串"},
        {"section_id": 3, "new_level": 1},                  # 重复
        {"section_id": 1, "new_level": 7},                  # 级别越界
        {"section_id": 1, "new_level": True},               # True 不算数字
        {"section_id": 2, "new_level": 1},                  # 没变化
        "不是对象",
    ]}
    fixes, problems = validate_fixes(data, sections)
    assert fixes == [SectionFix(3, 2, "数字写成了字符串")]
    assert len(problems) == 7


def test_missing_fixes_list_is_ignored(make_doc):
    fixes, problems = validate_fixes({"oops": []}, split_sections(make_doc(DOC)))
    assert fixes == [] and problems


def test_level_fix_moves_section_under_previous_one(make_doc):
    doc = make_doc(DOC)
    sections = split_sections(doc)
    fixed = apply_fixes(sections, [SectionFix(3, 2)])
    assert fixed[3].level == 2 and fixed[3].parent_id == 2
    assert sections[3].level == 1                            # 原列表不变，方便对比
    assert check_lossless(doc, fixed)


def test_not_a_heading_merges_back_without_losing_text(make_doc):
    doc = make_doc("# A\n内容一\n## 强调句\n内容二\n## 又一句\n内容三\n# B\n内容四")
    sections = split_sections(doc)
    sections[1].summary = "旧摘要"
    fixed = apply_fixes(sections, [SectionFix(2, None), SectionFix(3, None)])
    assert [s.title for s in fixed] == ["", "A", "B"]
    assert fixed[1].content == "内容一\n## 强调句\n内容二\n## 又一句\n内容三"
    assert fixed[1].summary is None                          # 内容变了，旧摘要作废
    assert check_lossless(doc, fixed)


def test_fix_structure_end_to_end_with_cache(make_doc, fake_llm, tmp_path):
    sections = split_sections(make_doc(DOC))
    llm = fake_llm({"fixes": [{"section_id": 3, "new_level": 2, "reason": "是题目2的代码"}]})
    cache = JsonCache(tmp_path / "fix.json")

    fixed, fixes, problems = fix_structure(sections, llm, cache)
    assert fixed[3].parent_id == 2 and problems == []
    assert "json" in llm.prompts[0]                           # JSON 模式要求提示词里出现 json

    again, _, _ = fix_structure(split_sections(make_doc(DOC)), fake_llm(), JsonCache(tmp_path / "fix.json"))
    assert again[3].parent_id == 2                           # 第二次直接用缓存，不调用 LLM


def test_llm_failure_keeps_original_structure(make_doc, fake_llm, tmp_path):
    sections = split_sections(make_doc(DOC))
    fixed, fixes, problems = fix_structure(sections, fake_llm(LLMError("HTTP 401")), JsonCache(tmp_path / "f.json"))
    assert fixed is sections and fixes == []
    assert "保持原结构" in problems[0]
    assert not (tmp_path / "f.json").exists()               # 失败的结果不写缓存


def test_only_root_is_skipped(make_doc, fake_llm, tmp_path):
    _, _, problems = fix_structure(split_sections(make_doc("没有标题")), fake_llm(), JsonCache(tmp_path / "f.json"))
    assert "跳过" in problems[0]
