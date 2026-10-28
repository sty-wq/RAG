"""阶段二 第 ③④ 步（MVP 版）：章节 → chunk。"""
from __future__ import annotations

from chunker import split_sections
from chunks import check_chunks_lossless, make_chunks, split_blocks, split_long_text, split_section_text

CODE = "```cpp\n" + "\n".join(f"int x{i} = {i};" for i in range(80)) + "\n\n// 空行也在代码里\n```"


def test_short_section_is_one_chunk():
    assert split_section_text("短内容", limit=100) == ["短内容"]


def test_empty_section_makes_no_chunk(make_doc):
    doc = make_doc("# 父\n## 子\n内容")
    chunks = make_chunks(doc, split_sections(doc))
    assert len(chunks) == 1
    assert chunks[0].heading_path == "测试.md > 父 > 子"      # 父节没有内容，但出现在路径里


def test_code_block_is_never_split():
    blocks = split_blocks(f"段落一\n\n{CODE}\n\n段落二")
    assert [is_code for _, is_code in blocks] == [False, True, False]
    parts = split_section_text(f"段落一\n\n{CODE}\n\n段落二", limit=200)
    assert CODE in parts                                     # 超长的代码块整块保留


def test_long_section_respects_limit_except_code():
    text = "\n\n".join(f"第{i}段。" + "内容" * 30 for i in range(10)) + f"\n\n{CODE}"
    parts = split_section_text(text, limit=200)
    assert all(len(p) <= 200 or p == CODE for p in parts)
    assert len(parts) > 1


def test_long_paragraph_is_split_by_sentences():
    text = "这是一句话。" * 50                                  # 300 字，没有空行
    pieces = split_long_text(text, limit=100)
    assert all(len(p) <= 100 for p in pieces)
    assert all(p.endswith("。") for p in pieces)             # 在句号处切开
    assert "".join(pieces) == text


def test_chunks_are_lossless_and_ids_stable(make_doc):
    doc = make_doc("开头\n# A\n" + "很长的段落。" * 200 + f"\n\n{CODE}\n# B\n内容")
    sections = split_sections(doc)
    chunks = make_chunks(doc, sections, limit=300)
    assert check_chunks_lossless(sections, chunks)
    assert [c.order for c in chunks] == list(range(len(chunks)))
    assert len({c.id for c in chunks}) == len(chunks)
    again = make_chunks(doc, split_sections(doc), limit=300)
    assert [c.id for c in again] == [c.id for c in chunks]   # 同样的输入，id 每次都一样


def test_embed_text_has_heading_path_and_source(make_doc):
    doc = make_doc("# 题目4\n## 参考代码\n代码", name="题库.docx")
    chunk = make_chunks(doc, split_sections(doc))[0]
    assert chunk.embed_text == "题库.docx > 题目4 > 参考代码\n代码"
    assert chunk.source_path == "notes/题库.docx"
