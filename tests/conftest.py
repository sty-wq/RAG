"""测试共用的小工具。

所有测试都不调用任何真实的 API（DeepSeek、OCR 服务商），不需要 key，也不花钱。
需要大模型的地方，用下面的 FakeLLM 代替。
"""
from __future__ import annotations

import pytest

from main import Document


@pytest.fixture
def make_doc():
    """用一段文字快速造一个 Document。"""
    def _make(content: str, name: str = "测试.md") -> Document:
        return Document(id="doc-1", name=name, content=content, source_path=f"notes/{name}")
    return _make


class FakeLLM:
    """假的大模型客户端：按顺序返回预先准备好的结果，并记下收到的提示词。

    replies 里可以放：字符串（chat 的回答）、dict（chat_json 的回答）、
    或者一个异常对象（模拟调用失败）。
    """

    def __init__(self, *replies, model: str = "fake-model"):
        self.model = model
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.last_reasoning = None

    def _next(self, prompt: str):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def chat(self, prompt: str, **kwargs) -> str:
        return self._next(prompt)

    def chat_json(self, prompt: str, **kwargs) -> dict:
        return self._next(prompt)


@pytest.fixture
def fake_llm():
    return FakeLLM
