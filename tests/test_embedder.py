"""向量化客户端：用假的 httpx.post 测试，不需要 Ollama。"""
from __future__ import annotations

import pytest

import embedder as embedder_module
from embedder import Embedder, EmbeddingError


class FakeResponse:
    def __init__(self, status_code: int, data=None):
        self.status_code = status_code
        self.text = "error body"
        self._data = {"data": data or []}

    def json(self):
        return self._data


def fake_post(monkeypatch, *responses):
    queue, calls = list(responses), []

    def post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers})
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):                       # 根据请求内容现算一个返回
            return item(json)
        return item

    monkeypatch.setattr(embedder_module.httpx, "post", post)
    monkeypatch.setattr(embedder_module.time, "sleep", lambda seconds: None)
    return calls


def echo_vectors(body):
    """每条文字返回一个 [序号, 长度] 的二维"向量"，并且故意把顺序打乱。"""
    data = [{"index": i, "embedding": [float(i), float(len(t))]} for i, t in enumerate(body["input"])]
    return FakeResponse(200, list(reversed(data)))


def test_defaults_point_to_local_ollama(monkeypatch):
    for name in ("EMBED_BASE_URL", "EMBED_MODEL", "EMBED_API_KEY", "EMBED_BATCH_SIZE"):
        monkeypatch.delenv(name, raising=False)
    e = Embedder()
    assert e.base_url == "http://localhost:11434/v1"
    assert e.model == "qwen3-embedding:0.6b" and e.batch_size == 10


def test_config_comes_from_environment(monkeypatch):
    monkeypatch.setenv("EMBED_BASE_URL", "https://cloud.example.com/v1/")
    monkeypatch.setenv("EMBED_MODEL", "text-embedding-v4")
    monkeypatch.setenv("EMBED_API_KEY", "sk-test")
    calls = fake_post(monkeypatch, echo_vectors)
    Embedder().embed(["a"])
    assert calls[0]["url"] == "https://cloud.example.com/v1/embeddings"
    assert calls[0]["json"]["model"] == "text-embedding-v4"
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-test"


def test_batches_and_keeps_input_order(monkeypatch):
    calls = fake_post(monkeypatch, echo_vectors, echo_vectors, echo_vectors)
    texts = [f"第{i}条" + "x" * i for i in range(25)]
    vectors = Embedder(base_url="http://x", api_key="", batch_size=10).embed(texts)
    assert [len(c["json"]["input"]) for c in calls] == [10, 10, 5]
    assert [v[1] for v in vectors] == [float(len(t)) for t in texts]    # 顺序和输入一致
    assert "Authorization" not in calls[0]["headers"]                   # Ollama 不需要 key


def test_wrong_number_of_vectors_is_an_error(monkeypatch):
    fake_post(monkeypatch, FakeResponse(200, [{"index": 0, "embedding": [1.0]}]))
    with pytest.raises(EmbeddingError, match="只返回了 1"):
        Embedder(base_url="http://x", api_key="").embed(["a", "b"])


def test_retry_then_success(monkeypatch):
    calls = fake_post(monkeypatch, FakeResponse(503), embedder_module.httpx.TransportError("断了"),
                      echo_vectors)
    assert Embedder(base_url="http://x", api_key="").embed(["a"]) == [[0.0, 1.0]]
    assert len(calls) == 3


def test_clear_message_when_ollama_is_not_running(monkeypatch):
    error = embedder_module.httpx.TransportError("refused")
    fake_post(monkeypatch, error, error, error)
    with pytest.raises(EmbeddingError, match="Ollama"):
        Embedder(base_url="http://localhost:11434/v1", api_key="").embed(["a"])


def test_no_retry_when_model_is_missing(monkeypatch):
    calls = fake_post(monkeypatch, FakeResponse(404))
    with pytest.raises(EmbeddingError, match="404"):
        Embedder(base_url="http://x", api_key="").embed(["a"])
    assert len(calls) == 1
