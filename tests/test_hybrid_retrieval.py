"""混合召回与重排的控制实验：不请求模型，不依赖现有笔记库。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from answer import answer_question
from api import Deps, create_app
from evaluation.run_retrieval import evaluate, print_table
from fastapi.testclient import TestClient
from lexical import BM25Index, tokenize
from reranker import RerankError
from retrieval import RetrievalConfig, fuse_rrf, passes_score_gate, rerank_hits, retrieve
from search import run_search
from vector_store import SearchHit


def hit(key, score=0.8, text="正文", doc="a.md"):
    return SearchHit(score, key, doc, f"notes/{doc}", f"{doc} > 标题", 0, text)


class Store:
    def __init__(self, dense, lexical=()):
        self.dense, self.lexical = dense, list(lexical)
        self.calls = []

    def read_meta(self):
        return {"embed_model": "model"}

    def search(self, vector, k, doc_name=None):
        self.calls.append(("dense", k, doc_name))
        return [h for h in self.dense if doc_name is None or h.doc_name == doc_name][:k]

    def lexical_search(self, question, k, doc_name=None):
        self.calls.append(("bm25", k, doc_name))
        return [h for h in self.lexical if doc_name is None or h.doc_name == doc_name][:k]


class Embedder:
    model = "model"

    def embed_one(self, question):
        return [1.0]

    def embed(self, questions):
        return [[1.0] for _ in questions]


class Scorer:
    def __init__(self):
        self.received = []

    def score(self, question, documents):
        self.received.append((question, documents))
        return [0.2 if "充分证据" in doc else -0.8 for doc in documents]

    def describe(self):
        return {"backend": "test", "model": "controlled"}


def test_bm25_finds_chinese_terms_english_identifiers_and_normalizes_width():
    index = BM25Index(["苹果手机的接口", "红黑树和 requestId 幂等", "banana database"])
    assert index.search("红黑树", 1)[0][0] == 1
    assert index.search("ＲＥＱＵＥＳＴＩＤ", 1)[0][0] == 1
    assert index.search("not_in_corpus", 3) == []
    assert "request_id" in tokenize("request_id")
    assert "c++" in tokenize("C++")


def test_bm25_empty_query_no_matches_and_document_filter():
    index = BM25Index(["red apple", "red apple", "yellow banana"])
    assert index.search("", 3) == []
    assert index.search("apple", 3, allowed={2}) == []
    assert index.search("apple", 3, allowed={1})[0][0] == 1
    assert BM25Index([]).search("apple") == []


def test_rrf_preserves_bm25_only_hits_and_distinct_chunks_with_same_text():
    a, b, c = hit("a"), hit("b"), hit("c")
    results = fuse_rrf([a, b], [replace(c, score=5.0), replace(b, score=3.0)], 3)
    assert [h.chunk_id for h in results] == ["b", "a", "c"]
    assert results[0].dense_score == 0.8 and results[0].lexical_score == 3.0
    assert results[2].dense_score is None and results[2].lexical_score == 5.0
    assert all(h.score_type == "rrf" and h.rrf_score == h.score for h in results)
    assert a.score_type == "cosine" and b.lexical_score is None  # 不改原始候选


def test_rrf_duplicate_chunk_in_one_route_does_not_get_extra_votes():
    a, b = hit("a"), hit("b")
    assert len(fuse_rrf([a, a], [b], 3)) == 2
    assert fuse_rrf([a, a], [], 3)[0].score == pytest.approx(1 / 61)


def test_hybrid_applies_filter_to_both_routes_and_separates_candidate_budget():
    store = Store([hit("a"), hit("b", doc="other.md")], [hit("c")])
    results = retrieve("问题", [1.0], store, 1, config=RetrievalConfig("hybrid", 20), doc_name="a.md")
    assert len(results) == 1
    assert store.calls == [("dense", 20, "a.md"), ("bm25", 20, "a.md")]


def test_bm25_does_not_call_embedding_or_require_matching_embedding_model():
    embedder = SimpleNamespace(model="different")  # 故意没有 embed_one
    store = Store([], [replace(hit("a"), score_type="bm25")])
    results, error = run_search("关键词", embedder, store, config=RetrievalConfig("bm25"))
    assert error is None and len(results) == 1
    assert store.calls == [("bm25", 5, None)]


def test_dense_model_mismatch_still_refused_before_search():
    store = Store([hit("a")])
    results, error = run_search("问题", SimpleNamespace(model="wrong"), store,
                                config=RetrievalConfig("hybrid"))
    assert results == [] and "wrong" in error and store.calls == []


def test_reranker_promotes_relevant_low_dense_score_and_keeps_original_source():
    store = Store([hit("a", 0.9, "无关"), hit("b", 0.1, "充分证据", doc="gold.md")])
    scorer = Scorer()
    results = retrieve("问题", [1.0], store, 1, config=RetrievalConfig("dense", 50), reranker=scorer)
    assert store.calls == [("dense", 50, None)]
    assert len(scorer.received[0][1]) == 2
    assert results[0].chunk_id == "b" and results[0].text == "充分证据"
    assert results[0].source_path == "notes/gold.md"
    assert results[0].score == 0.2 and results[0].dense_score == 0.1
    assert results[0].rerank_score == 0.2 and results[0].score_type == "rerank"


@pytest.mark.parametrize("scores", [[0.1], [float("nan"), 0.1], [float("inf"), 0.1], ["bad", 0.1]])
def test_invalid_rerank_scores_fail_instead_of_silent_fallback(scores):
    with pytest.raises(RerankError):
        rerank_hits("问题", [hit("a"), hit("b")], SimpleNamespace(score=lambda *args: scores), 1)


def test_empty_candidates_do_not_call_reranker():
    assert rerank_hits("问题", [], SimpleNamespace(), 1) == []


def test_cosine_gate_is_not_reused_for_bm25_rrf_or_rerank():
    assert not passes_score_gate(hit("a", 0.1), 0.35)
    assert passes_score_gate(replace(hit("a", 0.01), score_type="rrf"), 0.35)
    assert passes_score_gate(replace(hit("a", -2.0), score_type="rerank"), 0.35)
    assert not passes_score_gate(replace(hit("a", 0.2), score_type="rerank"), 0.35,
                                  RetrievalConfig(min_rerank_score=0.5))


def test_answer_keeps_rrf_sources_even_when_rrf_scores_are_below_point35(fake_llm):
    store = Store([hit("a", text="充分证据")], [hit("a", 5.0, "充分证据")])
    result = answer_question("问题", Embedder(), store, fake_llm("依据资料[1]。"),
                             config=RetrievalConfig("hybrid"))
    assert result.found and result.sources[0].score < 0.35
    assert result.sources[0].score_type == "rrf" and result.sources[0].cited


def test_api_uses_shared_hybrid_rerank_pipeline_and_allows_request_to_disable_it(fake_llm):
    store = Store([hit("a", text="无关"), hit("b", text="充分证据", doc="gold.md")], [hit("b", 5.0, "充分证据", "gold.md")])
    scorer = Scorer()
    deps = Deps(Embedder(), store, fake_llm("根据证据[1]。", "根据资料[1]。"), RetrievalConfig("hybrid", 20), scorer)
    with TestClient(create_app(deps)) as client:
        response = client.post("/ask", json={"question": "问题", "k": 1})
        assert response.status_code == 200
        source = response.json()["sources"][0]
        assert source["doc_name"] == "gold.md" and source["score_type"] == "rerank"
        assert source["rrf_score"] is not None and source["rerank_score"] == 0.2
        health = client.get("/health").json()
        assert health["retrieval"]["mode"] == "hybrid" and health["reranker"]["model"] == "controlled"
        response = client.post("/ask", json={"question": "问题", "k": 1, "mode": "dense", "rerank": False})
        assert response.status_code == 200 and response.json()["sources"][0]["score_type"] == "cosine"
    assert len(scorer.received) == 1


def test_api_reports_missing_reranker_and_validates_new_request_fields(fake_llm):
    with TestClient(create_app(Deps(Embedder(), Store([hit("a")]), fake_llm()))) as client:
        assert client.post("/ask", json={"question": "问题", "rerank": True}).status_code == 503
        for body in ({"mode": "bad"}, {"candidate_k": 0}, {"candidate_k": 1001}):
            assert client.post("/ask", json={"question": "问题", **body}).status_code == 422


def test_evaluation_uses_same_reranker_and_does_not_misreport_cosine_gate(capsys):
    questions = [{"qid": "q1", "type": "1doc", "question": "问题", "gold_docs": ["gold.md"]}]
    store = Store([hit("a", text="无关"), hit("b", text="充分证据", doc="gold.md")],
                  [hit("b", 5.0, "充分证据", "gold.md")])
    dense, _ = evaluate(questions, Embedder(), store, k_max=5)
    enhanced, details = evaluate(questions, Embedder(), store, k_max=5,
                                 config=RetrievalConfig("hybrid", 20), reranker=Scorer())
    assert dense["overall"]["hit@1"] == 0 and enhanced["overall"]["hit@1"] == 1
    assert enhanced["gate"]["enabled"] is False
    assert enhanced["gate"]["answerable_blocked"] is None
    assert details[0]["hits"][0]["chunk_id"] == "b"
    assert details[0]["hits"][0]["rrf_score"] is not None
    print_table(enhanced)
    assert "未配置拒答门槛" in capsys.readouterr().out


def test_bm25_evaluation_does_not_call_embedding():
    questions = [{"qid": "q1", "type": "1doc", "question": "问题", "gold_docs": ["a.md"]}]
    store = Store([], [replace(hit("a"), score_type="bm25")])
    summary, _ = evaluate(questions, object(), store, config=RetrievalConfig("bm25"))
    assert summary["overall"]["hit@1"] == 1


@pytest.mark.parametrize("kwargs", [{"mode": "bad"}, {"candidate_k": 0}, {"candidate_k": 1001},
                                    {"candidate_k": True}, {"rrf_k": 0}, {"min_rerank_score": float("nan")}])
def test_invalid_search_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        RetrievalConfig(**kwargs)
