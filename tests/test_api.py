"""问答接口：用 FastAPI 的 TestClient 发请求，依赖全部换成假的。"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from api import Deps, create_app  # noqa: E402
from llm import LLMError  # noqa: E402
from test_answer import HITS, FakeEmbedder, FakeStore  # noqa: E402


def client_with(llm, store=None) -> TestClient:
    return TestClient(create_app(Deps(embedder=FakeEmbedder(), store=store or FakeStore(HITS), llm=llm)))


def test_health(fake_llm):
    with client_with(fake_llm()) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok" and body["index"] == {"embed_model": "m"}
    assert body["retrieval"]["mode"] == "dense" and body["reranker"] is None


def test_ask_returns_answer_and_sources(fake_llm):
    with client_with(fake_llm("用红黑树[1]。")) as client:
        resp = client.post("/ask", json={"question": "红黑树怎么用", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True and body["answer"] == "用红黑树[1]。"
    assert body["sources"][0]["heading_path"] == "题库.docx > 题目3"
    assert body["sources"][0]["cited"] is True


@pytest.mark.parametrize("payload", [{"question": "   "}, {"question": "问题", "k": 0}, {}])
def test_bad_requests_are_rejected(fake_llm, payload):
    with client_with(fake_llm()) as client:
        assert client.post("/ask", json=payload).status_code == 422


def test_index_not_built_returns_503(fake_llm):
    with client_with(fake_llm(), store=FakeStore(HITS, meta={})) as client:
        resp = client.post("/ask", json={"question": "问题"})
    assert resp.status_code == 503 and "ingest" in resp.json()["detail"]


def test_llm_failure_returns_502(fake_llm):
    with client_with(fake_llm(LLMError("HTTP 401"))) as client:
        resp = client.post("/ask", json={"question": "红黑树怎么用"})
    assert resp.status_code == 502 and "401" in resp.json()["detail"]


def test_one_pass_reads_pool_once_and_preserves_citation_ids(fake_llm):
    from test_answer import hit

    hits = [hit(0.2, f"段落 {i}") for i in range(1, 51)]
    llm = fake_llm("两个模型在不同阶段引入信息[2][50]。")
    with client_with(llm, FakeStore(hits)) as client:
        response = client.post("/ask", json={"question": "比较两种方法", "one_pass": True, "candidate_k": 50})
    assert response.status_code == 200
    assert len(llm.prompts) == 1 and "段落 50" in llm.prompts[0]
    assert [s["index"] for s in response.json()["sources"] if s["cited"]] == [2, 50]


def test_one_pass_rejects_unknown_citation(fake_llm):
    with client_with(fake_llm("答案[99]")) as client:
        response = client.post("/ask", json={"question": "问题", "one_pass": True})
    assert response.status_code == 502
