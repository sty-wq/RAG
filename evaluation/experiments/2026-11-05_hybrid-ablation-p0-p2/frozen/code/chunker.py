"""阶段二：切分 —— 第 ② 步，把 Document 切成 Section（章节）。

设计见 DESIGN.md 4.3 节，要点：
- 每篇文档都有一个 level 0 的根节点，第一个标题之前的内容放在这里
- content 只放本节自己的原文，不含子节 → 每段原文只属于一个 Section
- 只有 level 说了算，parent_id 由 assign_parents() 根据 level 计算
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from main import Document, clean_text, parser


@dataclass
class Section:
    id: int                         # 文档内的顺序编号：根节点为 0，标题依次为 1, 2, 3…
    doc_id: str                     # 属于哪个文档
    title: str                      # 标题文字（根节点为空）
    level: int                      # 几级标题（根节点为 0）← 唯一说了算的层级信息
    content: str = ""               # 本节自己的原文，不含子节
    parent_id: int | None = None    # 上级是谁，由 assign_parents() 计算
    is_candidate: bool = False      # 是否是规则猜出来的疑似标题（下一步再用）
    summary: str | None = None      # 摘要（还没生成时为 None）


# 1~6 个 #，后面至少一个空格，再跟标题文字。
# 例："### 题目 1" 能匹配；"#include <iostream>" 不能（# 后面没有空格）
HEADING_RE = re.compile(r"(#{1,6})[ \t]+(.+)")


def is_fence(line: str) -> bool:
    """是不是代码块的开始/结束标记（``` 或 ~~~，前面允许有缩进）。"""
    stripped = line.lstrip()
    return stripped.startswith("```") or stripped.startswith("~~~")


def _join_lines(lines: list[str]) -> str:
    """把若干行拼回一段文字，只去掉首尾的空白行。

    不能直接用 .strip()：如果本节第一行是一行缩进的代码，
    strip() 会把它的缩进也删掉，代码就被改坏了。
    """
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return "\n".join(lines[start:end])


def split_sections(doc: Document) -> list[Section]:
    """按 Markdown 标题把一篇文档切成 Section 列表（按原文顺序）。

    和 docx_parser 是同一个套路：一行一行往下读，用 in_code 记住
    "现在是不是在代码块里"，用 current_lines 攒当前章节的内容行。
    注意判断顺序：先判断是不是在代码块里，再判断是不是标题，
    否则代码里的 "# 注释" 会被当成标题。
    """
    root = Section(id=0, doc_id=doc.id, title="", level=0)
    sections = [root]
    current = root          # 正在往里放内容的章节
    current_lines = []      # 当前章节攒下的内容行
    in_code = False

    for line in doc.content.split("\n"):
        # ① 代码块的开始或结束标记：切换状态，这一行本身也是内容
        if is_fence(line):
            in_code = not in_code
            current_lines.append(line)
            continue

        # ② 不在代码块里，并且是标题 → 收尾当前章节，开始一个新章节
        match = None if in_code else HEADING_RE.match(line)
        if match:
            current.content = _join_lines(current_lines)
            current = Section(
                id=len(sections),               # 顺序编号：第几个章节
                doc_id=doc.id,
                title=match.group(2).strip(),
                level=len(match.group(1)),      # "###" → 3
            )
            sections.append(current)
            current_lines = []
            continue

        # ③ 其他情况：普通内容行
        current_lines.append(line)

    current.content = _join_lines(current_lines)   # 最后一个章节也要收尾
    assign_parents(sections)
    return sections


def assign_parents(sections: list[Section]) -> None:
    """根据 level 计算每个章节的 parent_id（直接修改传进来的列表）。

    用一个栈记住"从根节点到当前位置的那条路径"：
    遇到一个新章节时，把栈顶所有 level >= 它的章节弹出去，
    剩下的栈顶就是它的上级；然后把它自己压进栈。

    写成独立函数，是因为应用 LLM 的层级修正之后，还要再调用一次。
    """
    stack: list[Section] = []
    for sec in sections:
        while stack and stack[-1].level >= sec.level:
            stack.pop()
        sec.parent_id = stack[-1].id if stack else None   # 根节点没有上级
        stack.append(sec)


def print_tree(sections: list[Section]) -> None:
    """把章节按树的样子打印出来，方便检查。"""
    depth = {}
    for sec in sections:
        depth[sec.id] = 0 if sec.parent_id is None else depth[sec.parent_id] + 1
        indent = "    " * depth[sec.id]
        title = sec.title or "(根节点)"
        print(f"{indent}[{sec.id}] L{sec.level} {title}"
              f"  —— {len(sec.content)} 字, parent={sec.parent_id}")


def _normalized_lines(text: str) -> list[str]:
    """去掉空白行；每行去掉首尾空白和开头的 #。

    为什么要去掉开头的 #：标题是几级（几个 #）属于"结构"，不属于"文字"。
    LLM 把 ### 改成 #### 之后，文字一个字都没变，检查应该照样通过。
    """
    result = []
    for line in text.split("\n"):
        if line.strip():
            result.append(line.strip().lstrip("#").strip())
    return result


def check_lossless(doc: Document, sections: list[Section]) -> bool:
    """检查切分（以及结构修正）前后，原文的文字一行都没丢、没多（空白行除外）。"""
    rebuilt = []
    for sec in sections:
        if sec.level > 0:
            rebuilt.append(sec.title)
        rebuilt.extend(_normalized_lines(sec.content))
    return _normalized_lines(doc.content) == rebuilt


def load_one(path_str: str) -> Document:
    """解析单个文件（和 load_directory 一样会做 clean_text），测试用。"""
    path = Path(path_str)
    doc = parser[path.suffix.lower()](path)
    doc.content = clean_text(doc.content)
    return doc


if __name__ == "__main__":
    for file in [
        "sample_notes/柠檬微趣CPP.docx",
        "sample_notes/embedding选型.md",
        "sample_notes/RAG项目笔记.md",
        "sample_notes/会议记录.txt",
    ]:
        doc = load_one(file)
        sections = split_sections(doc)
        print(f"===== {doc.name}：{len(sections)} 个章节，"
              f"原文完整：{check_lossless(doc, sections)} =====")
        print_tree(sections)
        print()
