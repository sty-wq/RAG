"""阶段二 第 ③④ 步（MVP 版）：把章节切成 chunk。

规则（DESIGN.md 4.4 节的硬性规则，MVP 只做最简单的版本）：
- 内容为空的章节（只有标题）不产生 chunk，但它的标题会出现在子节的标题路径里
- 章节没超过 MAX_CHUNK_CHARS：整节就是一块
- 超长的章节：先切成"段落"和"代码块"，再把相邻的合并到不超过上限
  - 代码块不拆，就算超长也单独成一块（以后用摘要索引处理，见 DESIGN.md 4.7）
  - 超长的普通段落按句子切，单句还超长就硬切
- 每块带两份文字：text 是原文（检索命中后交给 LLM），embed_text 是"标题路径 + 原文"（拿去向量化）
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from chunker import Section, is_fence, load_one, split_sections
from main import Document

MAX_CHUNK_CHARS = 800      # 📌 暂定：一块最多多少字（单个代码块除外），以后用评测数据来定
EMBED_MAX_CHARS = 4000     # 📌 暂定：拿去向量化的文字最多多少字，超长代码块会被截断


@dataclass
class Chunk:
    id: str             # 由 文档id + 章节id + 第几块 算出来，同一份结构每次都一样
    doc_id: str         # 属于哪个文档
    doc_name: str       # 文件名（出处）
    source_path: str    # 来源路径（出处）。Milvus 没有 JOIN，回答时要用的信息直接存在 chunk 上
    section_id: int     # 来自哪个章节
    heading_path: str   # 标题路径，比如 "柠檬微趣CPP.docx > 题目4 > 参考代码"
    order: int          # 在整篇文档里是第几块（从 0 开始）
    text: str           # 原文
    embed_text: str     # 拿去向量化的文字：标题路径 + 原文


def heading_path(sections: list[Section], sec: Section, doc_name: str) -> str:
    """从文件名开始，顺着 parent_id 往上找，拼出 "文件名 > 一级 > 二级 > …"。"""
    by_id = {s.id: s for s in sections}
    titles = []
    node = sec
    while node is not None and node.level > 0:      # 根节点没有标题，不放进路径
        titles.append(node.title)
        node = by_id.get(node.parent_id) if node.parent_id is not None else None
    return " > ".join([doc_name, *reversed(titles)])


def split_blocks(content: str) -> list[tuple[str, bool]]:
    """把一节内容切成 (文字, 是不是代码块) 的列表：空行分段，代码块整块保留。"""
    blocks: list[tuple[str, bool]] = []
    current: list[str] = []
    in_code = False

    def flush(is_code: bool) -> None:
        nonlocal current
        if any(line.strip() for line in current):
            blocks.append(("\n".join(current), is_code))
        current = []

    for line in content.split("\n"):
        if is_fence(line):
            if not in_code:              # 代码块开始：先把前面的段落收尾
                flush(False)
                current = [line]
                in_code = True
            else:                        # 代码块结束：整块收进去
                current.append(line)
                flush(True)
                in_code = False
            continue
        if in_code:
            current.append(line)         # 代码块里的空行也要保留
        elif line.strip():
            current.append(line)
        else:
            flush(False)                 # 空行 = 段落结束
    flush(in_code)                       # 没闭合的代码块，也按代码块处理
    return blocks


_SENTENCE_END = re.compile(r"(?<=[。！？!?；;\n])")


def split_long_text(text: str, limit: int) -> list[str]:
    """把超长的普通段落按句子切开，再合并到不超过 limit；单句还超长就硬切。"""
    pieces = []
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > limit:
            pieces.append(sentence[:limit])
            sentence = sentence[limit:]
        if sentence:
            pieces.append(sentence)

    merged, buf = [], ""
    for piece in pieces:
        if buf and len(buf) + len(piece) > limit:
            merged.append(buf)
            buf = ""
        buf += piece
    if buf:
        merged.append(buf)
    return [m.strip() for m in merged if m.strip()]


def split_section_text(content: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """把一节的内容切成若干块文字。"""
    if len(content) <= limit:
        return [content] if content.strip() else []

    parts, buf = [], ""
    for text, is_code in split_blocks(content):
        if len(text) > limit:
            if buf:
                parts.append(buf)
                buf = ""
            if is_code:
                parts.append(text)                       # 代码块不拆
            else:
                parts.extend(split_long_text(text, limit))
            continue
        candidate = f"{buf}\n\n{text}" if buf else text
        if len(candidate) <= limit:
            buf = candidate
        else:
            parts.append(buf)
            buf = text
    if buf:
        parts.append(buf)
    return parts


def make_chunks(doc: Document, sections: list[Section],
                limit: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """把一篇文档（已经切好、修正好的章节）变成 chunk 列表。"""
    chunks = []
    for sec in sections:
        path = heading_path(sections, sec, doc.name)
        for part, text in enumerate(split_section_text(sec.content, limit)):
            chunk_id = hashlib.md5(f"{doc.id}:{sec.id}:{part}".encode("utf-8")).hexdigest()
            chunks.append(Chunk(
                id=chunk_id,
                doc_id=doc.id,
                doc_name=doc.name,
                source_path=doc.source_path,
                section_id=sec.id,
                heading_path=path,
                order=len(chunks),
                text=text,
                embed_text=f"{path}\n{text}"[:EMBED_MAX_CHARS],
            ))
    return chunks


def check_chunks_lossless(sections: list[Section], chunks: list[Chunk]) -> bool:
    """检查切成 chunk 之后，每一节的文字一个字都没丢（忽略空白）。"""
    def squash(text: str) -> str:
        return "".join(text.split())

    for sec in sections:
        rebuilt = "".join(squash(c.text) for c in chunks if c.section_id == sec.id)
        if rebuilt != squash(sec.content):
            return False
    return True


if __name__ == "__main__":
    for file in ["sample_notes/柠檬微趣CPP.docx", "sample_notes/RAG项目笔记.md"]:
        doc = load_one(file)
        sections = split_sections(doc)
        chunks = make_chunks(doc, sections)
        print(f"===== {doc.name}：{len(chunks)} 块，"
              f"原文完整：{check_chunks_lossless(sections, chunks)} =====")
        for c in chunks:
            preview = c.text[:30].replace("\n", " ⏎ ")
            print(f"[{c.order}] {c.heading_path}  —— {len(c.text)} 字  {preview}…")
        print()
