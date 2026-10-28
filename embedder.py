"""阶段三：向量化 —— 把文字变成向量。

调用 OpenAI 兼容的 `POST /v1/embeddings` 接口。本地 Ollama、阿里云百炼、硅基流动都是这个接口，
所以代码只写一份，换平台只需要改环境变量：

  EMBED_BASE_URL    默认 http://localhost:11434/v1（本地 Ollama）
  EMBED_MODEL       默认 qwen3-embedding:0.6b
  EMBED_API_KEY     Ollama 不需要；云端平台需要
  EMBED_BATCH_SIZE  一次请求最多送多少条，默认 10（阿里云百炼的上限是 10）

⚠️ 换了模型，所有向量都要重新生成：不同模型生成的向量不能放在一起比较。
"""
from __future__ import annotations

import os
import time

import httpx

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class EmbeddingError(RuntimeError):
    """向量化失败。"""


class Embedder:
    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, batch_size: int | None = None, timeout: float = 120):
        self.base_url = (base_url or os.getenv("EMBED_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.model = model or os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
        self._api_key = api_key if api_key is not None else os.getenv("EMBED_API_KEY", "")
        self.batch_size = batch_size or int(os.getenv("EMBED_BATCH_SIZE", "10"))
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        """把一批文字变成向量，返回的顺序和输入一一对应。"""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start:start + self.batch_size]))
        if len({len(v) for v in vectors}) > 1:
            raise EmbeddingError("返回的向量维度不一致")
        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def _embed_batch(self, batch: list[str], retries: int = 2) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        body = {"model": self.model, "input": batch}
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

        for attempt in range(retries + 1):
            try:
                resp = httpx.post(url, json=body, headers=headers, timeout=self.timeout)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                problem = f"连不上 {url}（{type(exc).__name__}）。如果用的是本地 Ollama，确认它已经启动"
            else:
                if resp.status_code == 200:
                    data = resp.json()["data"]
                    if len(data) != len(batch):
                        raise EmbeddingError(f"送了 {len(batch)} 条，只返回了 {len(data)} 个向量")
                    # 接口返回的每一项带着 index，按它排序，保证和输入顺序一致
                    data = sorted(data, key=lambda item: item.get("index", 0))
                    return [item["embedding"] for item in data]
                if resp.status_code not in RETRYABLE_STATUS:
                    # 比如 404（模型名写错了、没有 pull）、401（key 不对），重试也没用
                    raise EmbeddingError(f"HTTP {resp.status_code}：{resp.text[:200]}")
                problem = f"HTTP {resp.status_code}"
            if attempt < retries:
                time.sleep(2 ** attempt)
        raise EmbeddingError(f"重试 {retries} 次后仍然失败：{problem}")
