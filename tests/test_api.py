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
    assert resp.json() == {"status": "ok", "index": {"embed_model": "m"}}


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
