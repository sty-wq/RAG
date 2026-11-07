"""重排客户端协议、失败边界和本地模型复用；默认不下载模型或访问网络。"""
import io
import json
import sys
from types import SimpleNamespace

import httpx
import pytest

import reranker as module
from _rerank_worker import serve
from reranker import HTTPReranker, LocalReranker, RerankError, create_reranker


def mock_http(monkeypatch, *responses):
    queue = list(responses)
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(module.httpx, "post", post)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    return calls


def response(rows, status=200):
    return httpx.Response(status, json={"results": rows})


def test_http_restores_original_document_order_from_response_indexes(monkeypatch):
    calls = mock_http(monkeypatch, response([{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": -1.0}]))
    client = HTTPReranker("https://example.test/v1/", "test-model", api_key="test-key")
    assert client.score("问题", ["不相关", "相关"]) == [-1.0, 0.9]
    url, req = calls[0]
    assert url == "https://example.test/v1/rerank"
    assert req["json"]["query"] == "问题" and req["json"]["top_n"] == 2
    assert req["headers"] == {"Authorization": "Bearer test-key", "Accept": "application/json"}
    assert "instruct" not in req["json"]
    assert "test-key" not in str(client.describe())


@pytest.mark.parametrize("rows", [[], [{"index": 0, "relevance_score": 1}] * 2,
                                    [{"index": True, "relevance_score": 1}],
                                    [{"index": 9, "relevance_score": 1}],
                                    [{"index": 0, "relevance_score": "nan"}],
                                    [{"index": 0}], None])
def test_http_rejects_missing_duplicate_out_of_range_or_nonfinite_scores(monkeypatch, rows):
    mock_http(monkeypatch, response(rows))
    with pytest.raises(RerankError, match="格式无效"):
        HTTPReranker("http://example.test", "m").score("q", ["a"])


def test_http_retries_temporary_failure_then_succeeds(monkeypatch):
    calls = mock_http(monkeypatch, response([], 503), httpx.TransportError("offline"),
                      response([{"index": 0, "relevance_score": 0.4}]))
    assert HTTPReranker("http://example.test", "m").score("q", ["a"]) == [0.4]
    assert len(calls) == 3


def test_http_auth_failure_is_not_retried_and_does_not_leak_response(monkeypatch):
    calls = mock_http(monkeypatch, httpx.Response(401, text="sensitive-upstream-details"))
    with pytest.raises(RerankError, match="HTTP 401") as error:
        HTTPReranker("http://example.test", "m").score("q", ["a"])
    assert len(calls) == 1 and "sensitive" not in str(error.value)


def test_http_exhausted_retries_are_reported(monkeypatch):
    mock_http(monkeypatch, httpx.TimeoutException("timeout"), httpx.TimeoutException("timeout"))
    with pytest.raises(RerankError, match="重试 1 次"):
        HTTPReranker("http://example.test", "m", retries=1).score("q", ["a"])


def test_worker_loads_model_once_and_receives_query_document_pairs(monkeypatch, tmp_path):
    loaded, pairs = [], []

    class Model:
        def __init__(self, name, **kwargs):
            loaded.append((name, kwargs))
            self.config = SimpleNamespace(_commit_hash="test-revision")

        def predict(self, values, **kwargs):
            pairs.append(values)
            return [0.8 for _ in values]

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=Model))
    requests = [
        {"action": "init", "model": "test-model", "device": "cpu", "max_length": 512,
         "cache_dir": str(tmp_path), "revision": None},
        {"action": "score", "question": "q", "documents": ["a", "b"], "batch_size": 2},
        {"action": "score", "question": "q2", "documents": ["c"], "batch_size": 2},
    ]
    output = io.StringIO()
    serve(io.StringIO("\n".join(json.dumps(r) for r in requests)), output)
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert responses[1]["scores"] == [0.8, 0.8] and responses[2]["scores"] == [0.8]
    assert responses[0]["revision"] == "test-revision"
    assert len(loaded) == 1 and loaded[0][1]["trust_remote_code"] is False
    assert pairs[0] == [("q", "a"), ("q", "b")]


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    path = tmp_path / "worker.py"
    path.write_text('''import json, os, sys, time
for line in sys.stdin:
    q = json.loads(line)
    if q["action"] == "init":
        r = {"ok": False, "error": "DEPENDENCY_MISSING"} if q["model"] == "missing" else {"ok": True, "revision": str(os.getpid())}
        if q["model"] == "qwen3-rerank":
            r = {"ok": False, "error": "RepositoryNotFoundError"}
    else:
        if q["question"] == "crash":
            sys.exit(7)
        if q["question"] == "slow":
            time.sleep(10)
        r = {"ok": True, "scores": [0.8] * len(q["documents"])}
    print(json.dumps(r), flush=True)
''')
    monkeypatch.setattr(module, "WORKER_SCRIPT", path)


def test_local_worker_is_lazy_reused_and_closed(fake_worker):
    scorer = LocalReranker("test-model")
    assert scorer._process is None and scorer.score("q", []) == []
    try:
        assert scorer.score("q", ["a", "b"]) == [0.8, 0.8]
        process = scorer._process
        assert scorer.score("q2", ["c"]) == [0.8]
        assert scorer._process is process and scorer.describe()["revision"] == str(process.pid)
    finally:
        scorer.close()
    assert process.poll() is not None and scorer._process is None


def test_missing_local_dependency_has_actionable_error(fake_worker):
    with pytest.raises(RerankError, match="requirements-rerank.txt"):
        LocalReranker("missing").score("q", ["a"])


def test_local_worker_crash_fails_request_and_next_request_can_restart(fake_worker):
    scorer = LocalReranker("test-model")
    with pytest.raises(RerankError, match="子进程已退出"):
        scorer.score("crash", ["a"])
    assert scorer._process is None
    try:
        assert scorer.score("q", ["a"]) == [0.8]
    finally:
        scorer.close()


def test_local_worker_timeout_terminates_process(fake_worker):
    scorer = LocalReranker("test-model", timeout=0.1)
    with pytest.raises(RerankError, match="超时"):
        scorer.score("slow", ["a"])
    assert scorer._process is None


def test_no_reranker_is_created_unless_enabled(monkeypatch):
    monkeypatch.delenv("RERANK_BACKEND", raising=False)
    assert create_reranker() is None
    assert isinstance(create_reranker("local"), LocalReranker)
    with pytest.raises(RerankError):
        create_reranker("unknown")


def test_warmup_loads_once_and_score_reuses_the_worker(fake_worker):
    scorer = LocalReranker('test-model')
    try:
        scorer.warmup()
        process = scorer._process
        scorer.warmup()
        assert scorer._process is process
        assert scorer.score('question', ['document']) == [0.8]
        assert scorer._process is process
    finally:
        scorer.close()


def test_bad_model_names_report_actual_config_and_cleanup(fake_worker, monkeypatch, tmp_path):
    monkeypatch.setenv('RERANK_MODEL', 'qwen3-rerank')
    scorer = LocalReranker(revision='test-revision', cache_dir=str(tmp_path))
    with pytest.raises(RerankError) as error:
        scorer.warmup()
    message = str(error.value)
    assert all(value in message for value in ['RepositoryNotFoundError', 'qwen3-rerank',
                                              'test-revision', str(tmp_path), 'Hugging Face', module.DEFAULT_MODEL])
    assert scorer._process is None


def test_qwen_http_sends_instruction_and_maps_all_50_scores(monkeypatch):
    monkeypatch.delenv('RERANK_INSTRUCT', raising=False)
    monkeypatch.setenv('RERANK_MODEL', 'qwen3-rerank')
    monkeypatch.setenv('RERANK_BASE_URL', 'https://api.quickrouter.ai/v1')
    monkeypatch.setenv('RERANK_API_KEY', 'test-key')
    calls = mock_http(monkeypatch, response([
        {'index': i, 'relevance_score': i / 50} for i in reversed(range(50))
    ]))
    scorer = create_reranker('http')
    docs = [f'passage {i}' for i in range(50)]
    assert scorer.score('question', docs) == [i / 50 for i in range(50)]
    url, request = calls[0]
    assert url == 'https://api.quickrouter.ai/v1/rerank'
    assert request['json'] == dict(model='qwen3-rerank', query='question', documents=docs,
                                  top_n=50, return_documents=False,
                                  instruct=HTTPReranker.QWEN_INSTRUCT)
    assert scorer.describe()['instruct'] == HTTPReranker.QWEN_INSTRUCT
    assert 'test-key' not in str(scorer.describe())


def test_http_instruction_configuration_precedence(monkeypatch):
    monkeypatch.setenv('RERANK_INSTRUCT', 'Environment instruction')
    scorer = HTTPReranker('https://example.test/v1', 'qwen3-rerank')
    assert scorer.instruct == 'Environment instruction'
    assert HTTPReranker('https://example.test/v1', 'qwen3-rerank',
                        instruct='Explicit instruction').instruct == 'Explicit instruction'
    with pytest.raises(ValueError, match='不能是空字符串'):
        HTTPReranker('https://example.test/v1', 'qwen3-rerank', instruct=' ')
