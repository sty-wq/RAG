"""阶段二 第 ②' 步：让 LLM 看大纲，修正标题层级。

设计见 DESIGN.md 4.3 节，要点：
- LLM 只看大纲（不看全文），只输出需要修改的章节（SectionFix），不改写原文
- 程序检查 LLM 的输出是否合法，不合法的那条丢掉并说明原因
- 只有 level 说了算：改完 level 后重新调用 assign_parents()
- LLM 说"不是标题"的，把标题这一行原样并回上一节，原文一个字都不丢
- 调用失败时退回原结构（规则的结果），不影响后续流程
- 用大纲的哈希做缓存：大纲没变就不再调用 LLM，结果也稳定
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

from chunker import Section, assign_parents, check_lossless, load_one, print_tree, split_sections
from llm import DeepSeekClient, LLMError
from outline import CACHE_DIR, SUMMARY_CACHE_FILE, JsonCache, build_outline, summarize_sections

PROMPT_VERSION = "v1"     # 改了提示词就改这里，旧缓存自动失效
USE_THINKING = True       # 实测：关掉时漏掉了"参考代码"的层级错误，打开后改对了；JSON 输出照样可用
FIX_CACHE_FILE = CACHE_DIR / "structure_fixes.json"

# 注意：示例里故意没有用测试文档里的真实情况（比如"参考代码"）。
# 如果提示词里直接写了答案，测试结果就不能说明模型真的会判断了。
FIX_PROMPT = """你是文档结构校对员。下面是一篇文档的大纲，由程序根据标题样式自动提取，标题的层级可能有错。

【大纲格式】
每行一个章节：[编号] L级别 标题 | 字数 | 摘要 / 原文 / 无内容
缩进表示目前的上下级关系。[0] 是根节点，放的是第一个标题之前的内容。
标着"（疑似标题）"的行，是程序猜出来的，需要你确认它到底是不是标题。

【常见错误】
1. 层级标错：某一节明明是前一节的一部分（比如补充说明、示例、答案），却被标成了和前一节同级。
2. 不是标题：一句正文或强调的话被设成了标题样式。
3. 疑似标题：判断它是不是真的标题；是的话给出正确级别，不是的话标为 null。

【规则】
- 只能修改标题的级别，不能调整章节顺序，不能修改根节点 [0]。
- new_level 填 1 到 6 的整数；如果你判断它不是标题，填 null。
- 只列出需要修改的章节。没列出的章节保持原样；结构没有问题就返回空列表。
- 拿不准的不要改。
- reason 用一句中文说明理由。

【输出】只输出 json，格式如下：
{{"fixes": [{{"section_id": 7, "new_level": 3, "reason": "第7节是第6节的补充说明，应该是第6节的下一级"}}]}}

【大纲】
{outline}"""


@dataclass
class SectionFix:
    section_id: int             # 改哪一节
    new_level: int | None       # 改成几级；None 表示"不是标题"
    reason: str = ""            # LLM 给出的理由


def _as_int(value):
    """LLM 有时会把数字写成字符串 "5"，宽容地转一下；转不了就返回原值。"""
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return value


def validate_fixes(data: dict, sections: list[Section]) -> tuple[list[SectionFix], list[str]]:
    """检查 LLM 返回的修正是否合法。返回 (合法的修正, 问题说明)。"""
    by_id = {sec.id: sec for sec in sections}
    items = data.get("fixes")
    if not isinstance(items, list):
        return [], ["返回的 JSON 里没有 fixes 列表，忽略整个结果"]

    fixes, problems, seen = [], [], set()
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict):
            problems.append(f"第 {i} 条不是一个对象，丢弃")
            continue
        sid = _as_int(item.get("section_id"))
        level = _as_int(item.get("new_level"))
        reason = str(item.get("reason", ""))

        # 注意：Python 里 True/False 也算 int，要单独排除
        if not isinstance(sid, int) or isinstance(sid, bool) or sid not in by_id:
            problems.append(f"第 {i} 条：section_id={sid!r} 不存在，丢弃")
        elif sid == 0:
            problems.append(f"第 {i} 条：根节点不能修改，丢弃")
        elif sid in seen:
            problems.append(f"第 {i} 条：第 {sid} 节被重复修改，只保留第一条")
        elif level is not None and (not isinstance(level, int) or isinstance(level, bool)
                                    or not 1 <= level <= 6):
            problems.append(f"第 {i} 条：new_level={level!r} 不合法，丢弃")
        elif level == by_id[sid].level:
            problems.append(f"第 {i} 条：第 {sid} 节的级别没有变化，忽略")
        else:
            fixes.append(SectionFix(sid, level, reason))
            seen.add(sid)
    return fixes, problems


def apply_fixes(sections: list[Section], fixes: list[SectionFix]) -> list[Section]:
    """按修正生成一份新的章节列表（不改动传进来的列表，方便对比修正前后）。"""
    fix_by_id = {fix.section_id: fix for fix in fixes}
    result: list[Section] = []
    for sec in sections:
        fix = fix_by_id.get(sec.id)
        if fix is None:
            result.append(replace(sec))                     # 复制一份，原样保留
        elif fix.new_level is None:
            # 不是标题：把标题这一行原样并回上一节（根节点不能改，所以 result 一定不为空）
            prev = result[-1]
            heading_line = sec.title if sec.is_candidate else "#" * sec.level + " " + sec.title
            prev.content = "\n".join(part for part in (prev.content, heading_line, sec.content) if part)
            prev.summary = None                             # 内容变了，旧摘要作废
        else:
            result.append(replace(sec, level=fix.new_level, is_candidate=False))
    assign_parents(result)                                  # 只有 level 说了算，上下级重新算
    return result


def fix_structure(sections: list[Section], client: DeepSeekClient,
                  cache: JsonCache) -> tuple[list[Section], list[SectionFix], list[str]]:
    """②' 的完整流程：生成大纲 → 问 LLM（有缓存先用缓存）→ 检查 → 应用。

    返回 (修正后的章节, 采用的修正, 问题说明)。
    """
    if len(sections) <= 1:
        return sections, [], ["只有根节点，没有标题可以修正，跳过"]

    outline_text = build_outline(sections)
    raw = f"{PROMPT_VERSION}|{client.model}|thinking={USE_THINKING}\n{outline_text}"
    key = hashlib.md5(raw.encode("utf-8")).hexdigest()

    cached = cache.get(key)
    if cached is not None:
        data = json.loads(cached)
    else:
        try:
            # 思考过程也算输出 token，给足额度，免得 JSON 被截断
            data = client.chat_json(FIX_PROMPT.format(outline=outline_text), thinking=USE_THINKING,
                                    max_tokens=8000 if USE_THINKING else 800)
        except LLMError as exc:
            # 退回规则的结果：结构不修正，但流程照样往下走
            return sections, [], [f"LLM 调用失败，保持原结构：{exc}"]
        cache.put(key, json.dumps(data, ensure_ascii=False))

    fixes, problems = validate_fixes(data, sections)
    return apply_fixes(sections, fixes), fixes, problems


if __name__ == "__main__":
    client = DeepSeekClient()
    summary_cache = JsonCache(SUMMARY_CACHE_FILE)
    fix_cache = JsonCache(FIX_CACHE_FILE)
    print(f"使用模型：{client.model}\n")

    for file in [
        "sample_notes/柠檬微趣CPP.docx",
        "sample_notes/embedding选型.md",
        "sample_notes/RAG项目笔记.md",
    ]:
        doc = load_one(file)
        sections = split_sections(doc)
        summarize_sections(sections, client, summary_cache)
        new_sections, fixes, problems = fix_structure(sections, client, fix_cache)

        print(f"===== {doc.name} =====")
        print("修正前：")
        print_tree(sections)
        if client.last_reasoning:
            # 只有这次真的调用了 LLM（不是从缓存拿的）并且开了思考模式，才会有
            print("LLM 的思考过程：")
            print("  " + client.last_reasoning.strip().replace("\n", "\n  "))
            client.last_reasoning = None
        print("LLM 的修正：" if fixes else "LLM 的修正：无")
        for fix in fixes:
            target = "不是标题" if fix.new_level is None else f"改成 L{fix.new_level}"
            print(f"  [{fix.section_id}] {target}：{fix.reason}")
        for problem in problems:
            print(f"  ⚠️ {problem}")
        if fixes:
            print("修正后：")
            print_tree(new_sections)
        print(f"原文完整：{check_lossless(doc, new_sections)}\n")

    print(client.usage_report())
