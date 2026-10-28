"""阶段二 第 ② 步：切章节、算上下级。"""
from __future__ import annotations

from chunker import assign_parents, check_lossless, split_sections

MD = """开头的内容
# 一
内容一
```python
# 代码里的注释，不是标题
x = 1
```
### 跳过了二级
内容二
## 二
~~~
## 也不是标题
~~~
"""


def test_split_sections_finds_headings_but_not_code_comments(make_doc):
    sections = split_sections(make_doc(MD))
    assert [(s.id, s.level, s.title) for s in sections] == [
        (0, 0, ""), (1, 1, "一"), (2, 3, "跳过了二级"), (3, 2, "二"),
    ]
    assert sections[0].content == "开头的内容"              # 第一个标题之前的内容放在根节点
    assert "# 代码里的注释" in sections[1].content           # 代码块原样留在内容里


def test_parents_follow_levels_even_when_levels_skip(make_doc):
    sections = split_sections(make_doc(MD))
    assert [s.parent_id for s in sections] == [None, 0, 1, 1]


def test_assign_parents_again_after_a_level_change(make_doc):
    sections = split_sections(make_doc("# A\n## B\n## C"))
    sections[2].level = 3                                     # 把 B 降一级
    assign_parents(sections)
    assert [s.parent_id for s in sections] == [None, 0, 1, 1]
    sections[3].level = 4                                     # 再把 C 降到 B 下面
    assign_parents(sections)
    assert sections[3].parent_id == 2


def test_document_without_headings_is_one_root_section(make_doc):
    sections = split_sections(make_doc("只有一段话\n\n第二段"))
    assert len(sections) == 1
    assert sections[0].content == "只有一段话\n\n第二段"


def test_indented_code_at_section_start_keeps_its_indent(make_doc):
    sections = split_sections(make_doc("# A\n    缩进的第一行\n    第二行"))
    assert sections[1].content.startswith("    缩进的第一行")


def test_split_is_lossless(make_doc):
    doc = make_doc(MD)
    assert check_lossless(doc, split_sections(doc))
