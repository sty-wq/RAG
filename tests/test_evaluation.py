"""评测代码本身也要测：指标算得对不对、抽样有没有按规则来。全部离线。"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from evaluation.build_crud import build, doc_file_name, snippets, squash
from evaluation.metrics import average, bigram_overlap, hit_at_k, recall_at_k, reciprocal_rank
from evaluation.rewrite_questions import clean, rewrite_all
from evaluation.run_retrieval import evaluate
from llm import LLMError
from outline import JsonCache

# ---------------- 指标 ----------------

RANKED = ["a", "x", "b", "a", "y", "c"]          # 第 i 个 chunk 来自哪篇文档（同一篇可以出现多次）


def test_hit_at_k():
    assert hit_at_k(RANKED, {"b"}, 2) == 0.0
    assert hit_at_k(RANKED, {"b"}, 3) == 1.0


def test_recall_at_k_counts_each_gold_doc_once():
    assert recall_at_k(RANKED, {"a", "b", "c"}, 4) == pytest.approx(2 / 3)    # a 出现两次也只算一篇
    assert recall_at_k(RANKED, {"a", "b", "c"}, 6) == 1.0
    assert recall_at_k(RANKED, set(), 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, {"b"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, {"zzz"}) == 0.0


def test_average():
    assert average([{"m": 1.0}, {"m": 0.0}], ["m"]) == {"m": 0.5}
    assert average([], ["m"]) == {"m": 0.0}


# ---------------- evaluate：用假的 embedding 和向量库 ----------------

class FakeEmbedder:
    def embed(self, texts):
        return [[float(i)] for i in range(len(texts))]


class FakeStore:
    """第 i 道题返回预先写好的结果。"""

    def __init__(self, results):
        self.results = results

    def search(self, vector, k):
        return [SimpleNamespace(doc_name=name, score=score) for name, score in self.results[int(vector[0])][:k]]


def test_evaluate_scores_and_gate():
    questions = [
        {"qid": "1doc-1", "type": "1doc", "question": "q1", "gold_docs": ["g1.txt"]},
        {"qid": "2docs-1", "type": "2docs", "question": "q2", "gold_docs": ["g2.txt", "g3.txt"]},
        {"qid": "un-1", "type": "unanswerable", "question": "q3", "gold_docs": []},
    ]
    store = FakeStore([
        [("g1.txt", 0.7), ("x.txt", 0.5)],               # 第 1 名就找对了
        [("x.txt", 0.6), ("g2.txt", 0.5)],               # 两篇只找到一篇，排第 2
        [("y.txt", 0.2)],                                # 没答案，最高分也很低 → 会被门槛拦下
    ])
    summary, details = evaluate(questions, FakeEmbedder(), store, k_max=10, min_score=0.35)

    assert summary["by_type"]["1doc"]["hit@1"] == 1.0
    assert summary["by_type"]["2docs"]["recall@5"] == 0.5
    assert summary["by_type"]["2docs"]["mrr"] == 0.5
    assert summary["overall"]["n"] == 2                  # 无答案题不算进检索指标
    assert summary["gate"]["answerable_blocked"] == 0.0
    assert summary["gate"]["unanswerable_blocked"] == 1.0
    assert "hit@1" not in details[2]


# ---------------- 抽样：用一个迷你版的 CRUD-RAG ----------------

def news(tag: str) -> str:
    return f"2023-08-01 10:00:00，正文：这是关于{tag}的新闻报道，" + f"{tag}的详细内容。" * 10


def make_fake_crud(root):
    def qa(event, count, tag):
        item = {"ID": event, "questions": f"关于{tag}的问题？", "answers": f"{tag}的答案"}
        for i in range(1, count + 1):
            item[f"news{i}"] = news(f"{tag}{i}")
        return item

    split = {
        # 事件 e1、e2 在三类题里都出现过，抽样时同一事件只能用一次
        "questanswer_1doc": [qa(f"e{i}", 1, f"一号{i}") for i in range(1, 7)],
        "questanswer_2docs": [qa(f"e{i}", 2, f"二号{i}") for i in range(1, 7)],
        "questanswer_3docs": [qa(f"e{i}", 3, f"三号{i}") for i in range(1, 7)],
    }
    (root / "data" / "crud_split").mkdir(parents=True)
    (root / "data" / "crud_split" / "split_merged.json").write_text(json.dumps(split, ensure_ascii=False),
                                                                    encoding="utf-8")
    corpus = [news(f"干扰{i}") for i in range(20)]
    corpus += [news(f"一号{i}1") for i in range(1, 7)]   # 标准答案的副本：必须被排除
    (root / "data" / "80000_docs").mkdir(parents=True)
    (root / "data" / "80000_docs" / "part_1").write_text("\n".join(corpus), encoding="utf-8")


def test_build_follows_the_sampling_rules(tmp_path):
    make_fake_crud(tmp_path / "crud")
    out = tmp_path / "out"
    manifest = build(tmp_path / "crud", out, per_task=1, unanswerable=1, distractors=5, seed=7)

    questions = [json.loads(line) for line in (out / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    corpus = {p.name: p.read_text(encoding="utf-8") for p in (out / "corpus").iterdir()}

    assert manifest["questions"] == {"1doc": 1, "2docs": 1, "3docs": 1, "unanswerable": 1}
    assert len({q["event_id"] for q in questions}) == 4                  # 同一事件只用一次
    assert [len(q["gold_docs"]) for q in questions] == [1, 2, 3, 0]
    for q in questions:
        assert all(name in corpus for name in q["gold_docs"])             # 标准答案都在语料里
    removed = questions[-1]["removed_doc"]
    assert removed not in corpus                                         # 无答案题的原文被拿掉了
    gold_flat = {squash(corpus[n]) for q in questions for n in q["gold_docs"]}
    distractors = [t for n, t in corpus.items() if squash(t) not in gold_flat]
    assert len(distractors) == 5
    guards = [s for q in questions for n in q["gold_docs"] for s in snippets(corpus[n])]
    assert not any(s in squash(t) for t in distractors for s in guards)   # 干扰项里没有标准答案的副本

    again = build(tmp_path / "crud", tmp_path / "out2", per_task=1, unanswerable=1, distractors=5, seed=7)
    assert again["questions"] == manifest["questions"]
    assert sorted(p.name for p in (tmp_path / "out2" / "corpus").iterdir()) == sorted(corpus)   # 同样的种子，结果一样


def test_doc_file_name_is_stable():
    assert doc_file_name("同一篇") == doc_file_name("同一篇")
    assert doc_file_name("同一篇") != doc_file_name("另一篇")


# ---------------- 难版题目：改写 ----------------

def test_bigram_overlap_ignores_punctuation():
    doc = "国家卫健委开展启明行动，防控儿童近视。"
    assert bigram_overlap("启明行动", doc) == 1.0
    assert bigram_overlap("启明行动？？", doc) == 1.0            # 标点不影响
    assert bigram_overlap("小孩近视怎么办", doc) < 0.5
    assert bigram_overlap("", doc) == 0.0


def test_clean_strips_prefix_quotes_and_extra_lines():
    assert clean("改写：“启明行动是干嘛的？”\n解释：……") == "启明行动是干嘛的？"
    assert clean("\n\n  问题：近视怎么防？ ") == "近视怎么防？"


QUESTIONS = [
    {"qid": "1doc-0001", "type": "1doc", "question": "国家卫生健康委员会于2023年7月28日启动的启明行动针对什么问题？",
     "gold_docs": ["a.txt"]},
    {"qid": "unanswerable-0001", "type": "unanswerable", "question": "某某活动于2023年8月1日在哪里举办？",
     "gold_docs": []},
]


def test_rewrite_keeps_labels_caches_and_survives_failures(fake_llm, tmp_path):
    cache = JsonCache(tmp_path / "rewrites.json")
    llm = fake_llm("改写：启明行动管什么的？", LLMError("HTTP 500"))

    hard, stats = rewrite_all(QUESTIONS, llm, cache, workers=1)

    assert hard[0]["question"] == "启明行动管什么的？"
    assert hard[0]["question_original"] == QUESTIONS[0]["question"]
    assert hard[0]["gold_docs"] == ["a.txt"]                   # 标准答案不变
    assert hard[1]["question"] == QUESTIONS[1]["question"]     # 失败的保留原题
    assert hard[1]["rewrite_failed"] is True
    assert stats["failed"] == 1 and stats["new_calls"] == 2

    # 再跑一次：成功的那道走缓存，只有失败的那道会重新请求
    llm2 = fake_llm("某活动在哪办的？")
    hard2, stats2 = rewrite_all(QUESTIONS, llm2, JsonCache(tmp_path / "rewrites.json"), workers=1)
    assert stats2["new_calls"] == 1 and len(llm2.prompts) == 1
    assert [h["question"] for h in hard2] == ["启明行动管什么的？", "某活动在哪办的？"]
    assert "国家卫生健康委员会" in llm.prompts[0]               # 改写时只给题目
    assert "a.txt" not in llm.prompts[0]
