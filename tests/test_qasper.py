"""Qasper 评测集：论文转 Markdown、题目分类、按证据段落算分。全部离线，不需要 pyarrow。"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from chunker import load_one, split_sections
from chunks import check_chunks_lossless, make_chunks
from evaluation.build_qasper import build, classify, paper_paragraphs, paper_to_markdown
from evaluation.contextualize_qasper import contextualize, with_title
from evaluation.metrics import covers, recall_at_k_covered
from evaluation.run_retrieval import evaluate
from llm import LLMError
from outline import JsonCache

LONG = "This paragraph explains the background of the model in great detail. " * 15   # 约 1000 字，会被切开


def answer(evidence=(), unanswerable=False, spans=(), yes_no=None, free=""):
    return {"unanswerable": unanswerable, "extractive_spans": list(spans), "yes_no": yes_no,
            "free_form_answer": free, "evidence": list(evidence), "highlighted_evidence": []}


PAPER = {
    "id": "2001.00001",
    "title": "A Tiny\nPaper",
    "abstract": "We study tiny papers.",
    "full_text": {
        "section_name": ["Introduction", "Model ::: Background", "Model ::: Encoder", "Results", None],
        "paragraphs": [
            ["Intro paragraph one.", "", "# hash at line start is not a heading"],
            [LONG],
            ["The encoder has 6 layers."],
            ["We reach 90% accuracy."],
            ["A paragraph without a section name."],
        ],
    },
    "qas": {
        "question": ["How many layers?", "What accuracy?", "What is in Table 3?", "Is it open source?",
                     "Who funded it?"],
        "answers": [
            {"answer": [answer(["The encoder has 6 layers."], spans=["6"]),
                        answer(["Model ::: Encoder", "The encoder has 6 layers.", "Intro paragraph one."],
                               free="six")]},
            {"answer": [answer(["We reach 90% accuracy."], spans=["90%"]), answer(unanswerable=True)]},
            {"answer": [answer(["FLOAT SELECTED: Table 3: Results."], free="BLEU 30")]},
            {"answer": [answer(unanswerable=True), answer(unanswerable=True)]},
            {"answer": [answer(free="NSF")]},
        ],
    },
}


# ---------------- 论文 → Markdown ----------------

def test_markdown_headings_levels_and_missing_parent():
    md = paper_to_markdown(PAPER)
    lines = md.splitlines()
    assert lines[0] == "# A Tiny Paper"                       # 标题里的换行压成空格
    assert "## Abstract" in lines and "## Introduction" in lines
    assert lines.count("## Model") == 1                       # 原数据里没有单独的 Model 章节，补一个，只补一次
    assert lines.index("## Model") < lines.index("### Background") < lines.index("### Encoder")
    assert "\\# hash at line start is not a heading" in lines  # 行首的 # 被转义
    assert "A paragraph without a section name." in md        # 没有章节名的段落接在上一个章节后面


def test_markdown_goes_through_our_pipeline(tmp_path):
    path = tmp_path / "paper.md"
    path.write_text(paper_to_markdown(PAPER), encoding="utf-8")
    doc = load_one(str(path))
    sections = split_sections(doc)
    titles = [(s.level, s.title) for s in sections if s.level > 0]
    assert titles == [(1, "A Tiny Paper"), (2, "Abstract"), (2, "Introduction"), (2, "Model"),
                      (3, "Background"), (3, "Encoder"), (2, "Results")]
    chunks = make_chunks(doc, sections)
    assert check_chunks_lossless(sections, chunks)
    encoder = next(c for c in chunks if "6 layers" in c.text)
    assert encoder.heading_path == "paper.md > A Tiny Paper > Model > Encoder"
    assert len([c for c in chunks if c.heading_path.endswith("Background")]) >= 2   # 长段落被切成了几块


# ---------------- 题目分类 ----------------

def test_classify_four_types():
    paragraphs = paper_paragraphs(PAPER)
    answers = [entry["answer"] for entry in PAPER["qas"]["answers"]]
    qtype, groups, dropped = classify(answers[0], paragraphs)
    assert qtype == "answerable" and dropped == 1               # 章节标题被当成证据的那条丢掉
    assert groups == [["The encoder has 6 layers."], ["The encoder has 6 layers.", "Intro paragraph one."]]
    assert classify(answers[1], paragraphs)[0] == "answerable"  # 有人说无答案、有人给了证据：按有证据算
    assert classify(answers[2], paragraphs)[0] == "float_only"
    assert classify(answers[3], paragraphs)[0] == "unanswerable"
    assert classify(answers[4], paragraphs)[0] == "no_evidence"


def test_build_writes_corpus_and_questions(tmp_path):
    out = tmp_path / "qasper"
    (out / "raw").mkdir(parents=True)
    (out / "raw" / "keep.parquet").write_text("x")
    manifest = build([PAPER], out)

    assert manifest["questions"] == {"answerable": 2, "float_only": 1, "unanswerable": 1, "no_evidence": 1}
    assert manifest["corpus_docs"] == 1 and manifest["dropped_non_paragraph_evidence"] == 1
    assert (out / "corpus" / "2001.00001.md").exists()
    assert (out / "raw" / "keep.parquet").exists()               # 下载的原始数据不能被删
    questions = [json.loads(x) for x in (out / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [q["should_answer"] for q in questions] == [True, True, False, False, True]
    assert questions[0]["answers"] == ["6", "six"]
    assert questions[3]["answers"] == ["Unanswerable", "Unanswerable"]


# ---------------- 按证据段落算分 ----------------

def test_covers_both_directions():
    assert covers("Intro.\n\nThe encoder has 6 layers.\n\nMore.", "The encoder has 6 layers.")   # 块里包含段落
    piece = LONG[:400]
    assert covers(piece, LONG)                                   # 块是长段落切出来的一部分
    assert not covers("layers.", "The encoder has 6 layers.")     # 太短的碎片不算
    assert not covers("Something else entirely.", "The encoder has 6 layers.")
    assert covers("\\# hash at line start", "# hash at line start")   # 转义过的段落照样能对上


def test_recall_takes_best_annotator():
    covered = [{0}, set(), {1}]
    groups = [{0, 1, 2, 3}, {0, 1}]
    assert recall_at_k_covered(covered, groups, 3) == 1.0        # 第二个标注者的两段全找到了
    assert recall_at_k_covered(covered, groups, 1) == 0.5


class TextStore:
    def __init__(self, results):
        self.results = results
        self.filters = []

    def search(self, vector, k, doc_name=None):
        self.filters.append(doc_name)
        rows = [r for r in self.results[int(vector[0])] if doc_name is None or r[0] == doc_name]
        return [SimpleNamespace(doc_name=d, text=t, score=s) for d, t, s in rows[:k]]


class CountingEmbedder:
    def embed(self, texts):
        return [[float(i)] for i in range(len(texts))]


def test_evaluate_by_evidence():
    questions = [
        {"qid": "p-q01", "type": "answerable", "question": "How many layers?", "gold_docs": ["p.md"],
         "evidence": [["The encoder has 6 layers."]], "should_answer": True},
        {"qid": "p-q02", "type": "float_only", "question": "Table 3?", "gold_docs": ["p.md"],
         "evidence": [], "should_answer": False},
        {"qid": "p-q03", "type": "no_evidence", "question": "Funding?", "gold_docs": ["p.md"],
         "evidence": [], "should_answer": True},
    ]
    store = TextStore([
        [("p.md", "Intro paragraph one.", 0.6), ("p.md", "The encoder has 6 layers.", 0.5)],  # 文档对了，第 2 块才是证据
        [("p.md", "Results.", 0.4)],
        [("p.md", "Intro paragraph one.", 0.5)],
    ])
    summary, details = evaluate(questions, CountingEmbedder(), store, k_max=10, min_score=0.45)

    row = details[0]
    assert row["hit@1"] == 0.0 and row["hit@3"] == 1.0 and row["mrr"] == 0.5
    assert row["doc_hit@1"] == 1.0                               # 按文档算第 1 名就对了，按证据算不对
    assert row["relevant_ranks"] == [2]
    assert summary["overall"]["n"] == 1                          # no_evidence 算不了分，也不算"该拒答"
    assert summary["counts"] == {"answerable": 1, "float_only": 1, "no_evidence": 1}
    assert summary["gate"]["n_unanswerable"] == 1
    assert summary["gate"]["unanswerable_blocked"] == 1.0
    assert "mrr" not in details[1] and "mrr" not in details[2]


def test_within_doc_only_searches_the_questions_paper():
    questions = [{"qid": "p-q01", "type": "answerable", "question": "How big is the dataset?",
                  "gold_docs": ["p.md"], "evidence": [["The dataset has 3k dialogs."]], "should_answer": True}]
    results = [[("other.md", "The dataset has 10k tweets.", 0.9), ("p.md", "The dataset has 3k dialogs.", 0.7)]]

    everywhere, _ = evaluate(questions, CountingEmbedder(), TextStore(results))
    store = TextStore(results)
    mine, _ = evaluate(questions, CountingEmbedder(), store, within_doc=True)

    assert everywhere["overall"]["hit@1"] == 0.0               # 别的论文也在说"数据集多大"，排到了前面
    assert mine["overall"]["hit@1"] == 1.0
    assert store.filters == ["p.md"]


# ---------------- 补上"是哪篇论文" ----------------

def test_with_title_keeps_acronyms():
    assert with_title("What datasets do they use?", "A Tiny Paper") == \
        'In the paper "A Tiny Paper", what datasets do they use?'
    assert with_title("BERT is used?", "T").startswith('In the paper "T", BERT')     # 缩写不改小写
    assert with_title("how big is it?", "T") == 'In the paper "T", how big is it?'


def test_contextualize_judges_without_title_and_caches(fake_llm, tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "p.md").write_text("# A Tiny Paper\n\n## Abstract\n\nx\n", encoding="utf-8")
    questions = [
        {"qid": "p-q01", "type": "answerable", "question": "What datasets do they use?", "gold_docs": ["p.md"]},
        {"qid": "p-q02", "type": "answerable", "question": "How big is the AntiScam corpus?", "gold_docs": ["p.md"]},
        {"qid": "p-q03", "type": "unanswerable", "question": "Who funded it?", "gold_docs": ["p.md"]},
    ]
    cache = JsonCache(tmp_path / "ctx.json")
    llm = fake_llm({"self_contained": False, "reason": "有代指"}, {"self_contained": True, "reason": "有专名"},
                   LLMError("HTTP 500"))
    items, stats = contextualize(questions, corpus, llm, cache, workers=1)

    assert items[0]["question"] == 'In the paper "A Tiny Paper", what datasets do they use?'
    assert items[0]["question_original"] == "What datasets do they use?"
    assert items[1]["question"] == "How big is the AntiScam corpus?"           # 能看出是哪篇：原样保留
    assert items[2]["judge_failed"] and items[2]["question"].startswith("In the paper")   # 失败：保守地补标题
    assert [i["group"] for i in items] == ["answerable:title", "answerable:self", "unanswerable:title"]
    assert stats["failed"] == 1
    assert all("A Tiny Paper" not in p for p in llm.prompts)                   # 判断时不给标题

    llm2 = fake_llm({"self_contained": False, "reason": "x"})
    contextualize(questions, corpus, llm2, JsonCache(tmp_path / "ctx.json"), workers=1)
    assert len(llm2.prompts) == 1                                              # 只有失败的那道重新请求


def test_scores_are_reported_by_group():
    base = {"type": "answerable", "gold_docs": ["p.md"], "evidence": [["The dataset has 3k dialogs."]],
            "should_answer": True}
    questions = [{**base, "qid": "a", "question": "q", "group": "answerable:self"},
                 {**base, "qid": "b", "question": "q", "group": "answerable:title"}]
    results = [[("p.md", "The dataset has 3k dialogs.", 0.9)], [("x.md", "nothing", 0.9)]]
    summary, _ = evaluate(questions, CountingEmbedder(), TextStore(results))
    assert summary["by_type"]["answerable:self"]["hit@1"] == 1.0
    assert summary["by_type"]["answerable:title"]["hit@1"] == 0.0


@pytest.mark.parametrize("k", [1, 5])
def test_crud_questions_still_scored_by_document(k):
    questions = [{"qid": "1doc-1", "type": "1doc", "question": "q", "gold_docs": ["g.txt"]}]
    store = TextStore([[("g.txt", "anything", 0.9)]])
    summary, _ = evaluate(questions, CountingEmbedder(), store)
    assert summary["overall"][f"hit@{k}"] == 1.0
