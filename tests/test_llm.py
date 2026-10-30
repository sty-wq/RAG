"""DeepSeek 客户端：用假的 httpx.post 测试，不发真实请求。"""
from __future__ import annotations

import pytest

import llm
from llm import DeepSeekClient, LLMError


class FakeResponse:
    def __init__(self, status_code: int, content: str | None = None, reasoning: str | None = None):
        self.status_code = status_code
        self.text = "error body"
        self._data = {
            "choices": [{"message": {"content": content, "reasoning_content": reasoning}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }

    def json(self):
        return self._data


@pytest.fixture
def client(tmp_path, monkeypatch):
    env = tmp_path / ".env.ocr"
    env.write_text("DEEPSEEK_BASE_URL=https://example.com/\nDEEPSEEK_API_KEY=sk-test\n"
                   "DEEPSEEK_MODEL=test-model\n", encoding="utf-8")
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)     # 重试时不真的等
    return DeepSeekClient(env)


def fake_post(monkeypatch, *responses):
    """让 httpx.post 依次返回 responses，并记下每次请求的内容。"""
    queue, calls = list(responses), []

    def post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers})
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(llm.httpx, "post", post)
    return calls


def test_missing_config_is_reported(tmp_path):
    env = tmp_path / ".env.ocr"
    env.write_text("DEEPSEEK_API_KEY=x\n", encoding="utf-8")
    with pytest.raises(LLMError, match="DEEPSEEK_BASE_URL"):
        DeepSeekClient(env)


def test_missing_config_file_is_reported_clearly(tmp_path):
    with pytest.raises(LLMError, match="不存在"):
        DeepSeekClient(tmp_path / "no-such.env")


def test_config_path_that_is_a_directory_is_reported_clearly(tmp_path):
    # 回归测试：docker run -v 的源文件不存在时，Docker 会建一个同名的空文件夹
    folder = tmp_path / ".env.ocr"
    folder.mkdir()
    with pytest.raises(LLMError, match="文件夹"):
        DeepSeekClient(folder)


def test_request_body_disables_thinking_by_default(client, monkeypatch):
    calls = fake_post(monkeypatch, FakeResponse(200, "你好"))
    assert client.chat("hi") == "你好"
    body = calls[0]["json"]
    assert calls[0]["url"] == "https://example.com/chat/completions"
    assert body["thinking"] == {"type": "disabled"} and body["temperature"] == 0
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-test"
    assert client.usage_report() == "共调用 1 次，输入 10 token，输出 2 token"


def test_thinking_mode_and_reasoning(client, monkeypatch):
    calls = fake_post(monkeypatch, FakeResponse(200, '{"ok": 1}', reasoning="先想一想"))
    assert client.chat_json("json", thinking=True) == {"ok": 1}
    body = calls[0]["json"]
    assert body["thinking"] == {"type": "enabled"} and "temperature" not in body
    assert body["response_format"] == {"type": "json_object"}
    assert client.last_reasoning == "先想一想"


def test_retries_on_server_and_network_errors(client, monkeypatch):
    calls = fake_post(monkeypatch, FakeResponse(503), llm.httpx.TransportError("断网"),
                      FakeResponse(200, "第三次成功"))
    assert client.chat("hi") == "第三次成功"
    assert len(calls) == 3


def test_no_retry_on_auth_error(client, monkeypatch):
    calls = fake_post(monkeypatch, FakeResponse(401))
    with pytest.raises(LLMError, match="401"):
        client.chat("hi")
    assert len(calls) == 1


def test_gives_up_after_retries(client, monkeypatch):
    fake_post(monkeypatch, FakeResponse(500), FakeResponse(500), FakeResponse(500))
    with pytest.raises(LLMError, match="重试 2 次"):
        client.chat("hi")


def test_json_mode_retries_empty_content(client, monkeypatch):
    fake_post(monkeypatch, FakeResponse(200, ""), FakeResponse(200, '{"fixes": []}'))
    assert client.chat_json("json") == {"fixes": []}


def test_chat_json_rejects_non_json(client, monkeypatch):
    fake_post(monkeypatch, FakeResponse(200, "不是 JSON"), FakeResponse(200, "还不是"))
    with pytest.raises(LLMError, match="不是合法 JSON"):
        client.chat_json("json")
