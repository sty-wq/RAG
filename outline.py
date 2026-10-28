"""阶段二 第 ② 步（后半）：给章节写摘要，生成给 LLM 看的大纲。

设计见 DESIGN.md 4.3 节：
- content 为空的章节（只有标题的父章节）：不总结，大纲里写"无内容"
- content 很短的章节：不总结，直接把原文放进大纲
- 其他章节：让模型用一句话概括"这段讲的是什么"（暂时用 DeepSeek，以后换成 4B 小模型）
- 用内容哈希做缓存：没改过的章节不会重复调用模型
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from chunker import Section, load_one, split_sections
from llm import DeepSeekClient, LLMError

SHORT_CONTENT = 100        # 📌 暂定：不到这么多字的章节不总结，直接放原文
MAX_SUMMARY_INPUT = 8000   # 📌 暂定：超长章节只取前这么多字去总结，控制成本
PROMPT_VERSION = "v1"      # 改了提示词就把这里改一下，旧的缓存会自动失效

CACHE_DIR = Path(__file__).resolve().parent / ".rag_cache"
SUMMARY_CACHE_FILE = CACHE_DIR / "summaries.json"

SUMMARY_PROMPT = """下面是一篇文档中的一个章节。请用一句话（不超过 50 字）概括它讲的是什么。
要求：只输出这句话本身，不要加"本章节"之类的前缀，不要重复标题。

章节标题：{title}
章节内容：
{content}"""


class JsonCache:
    """一个存在 JSON 文件里的小缓存：key → 字符串。

    摘要（②）和结构修正（②'）都用它，各存各的文件。
    key 都是"会影响结果的东西"拼起来再算哈希：内容改了，哈希就变了，
    自然会重新调用模型；没改的直接复用，不花钱。
    """

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, str] = {}
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def put(self, key: str, summary: str) -> None:
        self.data[key] = summary
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 每存一条就写一次文件：中途出错，前面花钱得到的结果也不会丢
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1),
                             encoding="utf-8")


def summary_key(model: str, sec: Section) -> str:
    """摘要缓存的 key：提示词版本 + 模型 + 标题 + 内容 的哈希。"""
    raw = f"{PROMPT_VERSION}|{model}|{sec.title}\n{sec.content}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def summarize_sections(sections: list[Section], client: DeepSeekClient,
                       cache: JsonCache) -> dict[str, int]:
    """给需要的章节填上 summary（直接修改传进来的列表），返回统计数字。"""
    stats = {"空": 0, "太短": 0, "缓存命中": 0, "新调用": 0, "失败": 0}
    for sec in sections:
        if not sec.content:
            stats["空"] += 1
            continue
        if len(sec.content) < SHORT_CONTENT:
            stats["太短"] += 1
            continue

        key = summary_key(client.model, sec)
        cached = cache.get(key)
        if cached is not None:
            sec.summary = cached
            stats["缓存命中"] += 1
            continue

        prompt = SUMMARY_PROMPT.format(title=sec.title or "（无标题）",
                                       content=sec.content[:MAX_SUMMARY_INPUT])
        try:
            sec.summary = client.chat(prompt, max_tokens=120)
        except LLMError as exc:
            # 一节失败不影响其他节：summary 留空，大纲里会退回显示开头一句
            print(f"  ⚠️ 第 {sec.id} 节「{sec.title}」总结失败：{exc}")
            stats["失败"] += 1
            continue
        cache.put(key, sec.summary)
        stats["新调用"] += 1
    return stats


def _one_line(text: str, limit: int) -> str:
    """把多行文字压成一行，太长就截断，大纲里每节只占一行。"""
    flat = " ⏎ ".join(line.strip() for line in text.split("\n") if line.strip())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def build_outline(sections: list[Section]) -> str:
    """生成给 LLM 看的大纲：每节一行，[编号] 级别 标题 | 字数 | 摘要或原文。"""
    depth: dict[int, int] = {}
    lines = []
    for sec in sections:
        depth[sec.id] = 0 if sec.parent_id is None else depth[sec.parent_id] + 1
        title = sec.title or "(根节点)"
        if sec.is_candidate:
            title += "（疑似标题）"

        if not sec.content:
            body = "（无内容）"
        elif sec.summary:
            body = f"摘要：{sec.summary}"
        elif len(sec.content) < SHORT_CONTENT:
            body = f"原文：{_one_line(sec.content, SHORT_CONTENT)}"
        else:
            body = f"开头：{_one_line(sec.content, 40)}"      # 总结失败时的退路

        indent = "  " * depth[sec.id]
        lines.append(f"{indent}[{sec.id}] L{sec.level} {title} | {len(sec.content)} 字 | {body}")
    return "\n".join(lines)


if __name__ == "__main__":
    client = DeepSeekClient()
    cache = JsonCache(SUMMARY_CACHE_FILE)
    print(f"使用模型：{client.model}\n")

    for file in [
        "sample_notes/柠檬微趣CPP.docx",
        "sample_notes/embedding选型.md",
        "sample_notes/RAG项目笔记.md",
    ]:
        doc = load_one(file)
        sections = split_sections(doc)
        stats = summarize_sections(sections, client, cache)
        print(f"===== {doc.name} =====  {stats}")
        print(build_outline(sections))
        print()

    print(client.usage_report())
