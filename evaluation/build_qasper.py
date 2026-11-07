"""把 Qasper（https://allenai.org/data/qasper）做成我们的评测集：测"在长文档里找到对的那一段"。

为什么要它：CRUD-RAG 的新闻太短，95% 的标准答案文档整篇就是一块，测不到切块。
Qasper 是 NLP 论文全文（中位数约 2 万字符、12 个章节），自带章节结构，
每道题都有人工标注的证据段落，可以按"块"判断找没找对。

做法：
1. 每篇论文转成一个 Markdown 文件：标题用 #，摘要和各章节用 ## / ###（子章节在原数据里写成
   "Model ::: Background"）。父章节自己没有正文时原数据里不会单独出现，这里补上它的标题
2. 所有论文放进同一个库：别的论文都是干扰项，更像真实的笔记库
3. 每道题分成四类：
   answerable   至少一个标注者给了文字证据 → 正常算检索分
   float_only   证据全是图表（"FLOAT SELECTED: Table 2 ..."）。数据里没有表格内容，
                我们答不了，应该拒答。以后能解析表格了，这类题会变成"应该能回答"
   unanswerable 所有标注者都说论文里没有答案 → 应该拒答
   no_evidence  有人给了答案但没标证据 → 算不了检索分，先留着，以后评生成质量用
4. 证据只保留和某个正文段落一字不差的（忽略空白）。标注者偶尔会把章节标题也选成证据，这些丢掉

用法（在项目根目录运行，需要 pyarrow）：
  python -m evaluation.build_qasper
  python -m evaluation.run_retrieval --data evaluation/data/qasper --label "qasper 基线"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

from evaluation.metrics import squash

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW = PROJECT_DIR / "evaluation" / "data" / "qasper" / "raw" / "test.parquet"
DEFAULT_OUT = PROJECT_DIR / "evaluation" / "data" / "qasper"
SECTION_SEP = ":::"
FLOAT_PREFIX = "FLOAT SELECTED"
TYPES = ("answerable", "float_only", "unanswerable", "no_evidence")

# 行首是 "# xxx" 的正文会被我们的 Markdown 解析器当成标题，前面加个反斜杠
_HEADING_LIKE = re.compile(r"^(\s*)(#{1,6}[ \t])")


def escape_paragraph(text: str) -> str:
    return "\n".join(_HEADING_LIKE.sub(r"\1\\\2", line) for line in text.split("\n"))


def one_line(text: str) -> str:
    return " ".join(text.split())


def paper_to_markdown(paper: dict) -> str:
    """一篇论文 → Markdown。标题是一级标题，摘要和章节从二级开始，最深到六级。"""
    lines = [f"# {one_line(paper['title'])}", "", "## Abstract", "", escape_paragraph(paper["abstract"].strip()), ""]
    opened: list[str] = []                  # 当前所在的章节路径，比如 ["Model", "Background"]
    full_text = paper["full_text"]
    for name, paragraphs in zip(full_text["section_name"], full_text["paragraphs"]):
        path = [one_line(part) for part in (name or "").split(SECTION_SEP) if part.strip()]
        if path:
            same = 0                         # 和上一个章节共同的前缀有几层，这几层的标题已经写过了
            while same < min(len(path), len(opened)) and path[same] == opened[same]:
                same += 1
            if same == len(path):            # 回到了上层章节（或重复出现），至少要重新写最后一层的标题
                same -= 1
            for depth in range(same, len(path)):
                lines += [f"{'#' * min(depth + 2, 6)} {path[depth]}", ""]
            opened = path
        for paragraph in paragraphs:         # 没有章节名的段落，接在上一个章节后面
            if paragraph.strip():
                lines += [escape_paragraph(paragraph.strip()), ""]
    return "\n".join(lines).rstrip() + "\n"


def paper_paragraphs(paper: dict) -> set[str]:
    """论文里所有正文段落（含摘要），去掉空白后的集合，用来核对证据。"""
    paragraphs = {squash(paper["abstract"])}
    for group in paper["full_text"]["paragraphs"]:
        paragraphs.update(squash(p) for p in group if p.strip())
    return paragraphs


def answer_text(answer: dict) -> str:
    if answer["unanswerable"]:
        return "Unanswerable"
    if answer["yes_no"] is not None:
        return "Yes" if answer["yes_no"] else "No"
    if answer["extractive_spans"]:
        return "; ".join(span.strip() for span in answer["extractive_spans"])
    return answer["free_form_answer"].strip()


def classify(answers: list[dict], paragraphs: set[str]) -> tuple[str, list[list[str]], int]:
    """按标注者给的证据给题目分类。返回 (类型, 每个标注者的证据段落, 丢掉的非正文证据条数)。"""
    groups, has_float, dropped = [], False, 0
    for answer in answers:
        evidence = []
        for text in answer["evidence"]:
            if text.startswith(FLOAT_PREFIX):
                has_float = True
            elif squash(text) in paragraphs:
                evidence.append(text)
            else:
                dropped += 1                 # 基本都是章节标题
        if evidence:
            groups.append(evidence)
    if groups:
        return "answerable", groups, dropped
    if has_float:
        return "float_only", [], dropped
    if all(answer["unanswerable"] for answer in answers):
        return "unanswerable", [], dropped
    return "no_evidence", [], dropped


def build(papers: list[dict], out_dir: Path, source_sha256: str = "") -> dict:
    questions, dropped_total = [], 0
    docs: dict[str, str] = {}
    for paper in papers:
        name = f"{paper['id']}.md"
        docs[name] = paper_to_markdown(paper)
        paragraphs = paper_paragraphs(paper)
        qas = paper["qas"]
        for i, (question, entry) in enumerate(zip(qas["question"], qas["answers"]), 1):
            answers = entry["answer"]
            qtype, groups, dropped = classify(answers, paragraphs)
            dropped_total += dropped
            questions.append({
                "qid": f"{paper['id']}-q{i:02d}", "type": qtype, "paper_id": paper["id"],
                "question": question.strip(),
                "answers": [answer_text(a) for a in answers],
                "gold_docs": [name],
                "evidence": groups,
                # 应不应该回答：图表题和无答案题都应该拒答；no_evidence 有答案，只是没标证据
                "should_answer": qtype in ("answerable", "no_evidence"),
            })

    if out_dir.exists():                     # 语料变了，旧的向量库也要删；raw/ 和 runs/ 不动
        for sub in ("corpus", "index"):
            shutil.rmtree(out_dir / sub, ignore_errors=True)
    corpus_out = out_dir / "corpus"
    corpus_out.mkdir(parents=True)
    for name, text in docs.items():
        (corpus_out / name).write_text(text, encoding="utf-8")
    with (out_dir / "questions.jsonl").open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    counts = Counter(q["type"] for q in questions)
    manifest = {
        "source": "Qasper test split (huggingface allenai/qasper, parquet)",
        "source_sha256": source_sha256,
        "seed": None,                        # 没有抽样，全部使用
        "params": {},
        "questions": {t: counts.get(t, 0) for t in TYPES},
        "corpus_docs": len(docs),
        "dropped_non_paragraph_evidence": dropped_total,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_parquet(path: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("需要 pyarrow：pip install pyarrow") from exc
    return pq.read_table(path).to_pylist()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把 Qasper 做成评测集")
    parser.add_argument("--raw", default=str(DEFAULT_RAW), help="下载好的 parquet 文件")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录")
    args = parser.parse_args(argv)

    raw = Path(args.raw)
    manifest = build(load_parquet(raw), Path(args.out), hashlib.sha256(raw.read_bytes()).hexdigest()[:16])
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\n已写入：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
