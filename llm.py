"""调用 DeepSeek 的最小客户端。

- 不依赖 openai 库，直接用项目里已经装好的 httpx 发 HTTP 请求，
  这样也能看清楚"调用大模型"其实就是一次 POST 请求。
- 配置从 .env.ocr 读取：DEEPSEEK_BASE_URL、DEEPSEEK_API_KEY、DEEPSEEK_MODEL。
- 写成一个类，摘要（②）、大纲校验（②'）、切分方案（⑤）都用它。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values

ENV_FILE = Path(__file__).resolve().parent / ".env.ocr"

# 遇到这些状态码可以稍等一下再试：429 = 请求太频繁，5xx = 服务器临时出错
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """调用大模型失败。"""


class DeepSeekClient:
    def __init__(self, env_file: Path = ENV_FILE, timeout: float = 60):
        # 先确认配置文件真的存在、而且是个文件。
        # 踩过的坑：docker run -v 挂载时源文件不存在，Docker 会建一个同名的空文件夹，
        # 如果不先检查，这里只会报"缺少配置"，让人误以为是文件内容写错了
        if not env_file.is_file():
            what = "是一个文件夹，不是文件" if env_file.is_dir() else "不存在"
            raise LLMError(f"找不到配置文件：{env_file} {what}。"
                           "如果是在 Docker 里运行，检查 -v 挂载的源路径对不对")
        cfg = dotenv_values(env_file)
        needed = ("DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL")
        missing = [key for key in needed if not cfg.get(key)]
        if missing:
            raise LLMError(f"{env_file.name} 里缺少配置：{', '.join(missing)}")
        self.base_url = cfg["DEEPSEEK_BASE_URL"].rstrip("/")
        self.model = cfg["DEEPSEEK_MODEL"]
        self._api_key = cfg["DEEPSEEK_API_KEY"]     # 只在发请求时用，不打印、不写日志
        self.timeout = timeout
        # 累计用量，方便心里有数花了多少 token
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.last_reasoning: str | None = None     # 思考模式下，最近一次的思考过程

    def chat(self, prompt: str, *, max_tokens: int = 300, retries: int = 2,
             json_mode: bool = False, thinking: bool = False) -> str:
        """发一条消息，返回模型的回答文字。失败时抛出 LLMError。

        Args:
            json_mode: 打开 DeepSeek 的 JSON 输出模式。提示词里必须出现 "json"
                这个词，并给出格式示例。
            thinking: 是否开启思考模式。DeepSeek V4 默认是开着的；摘要这种简单
                任务用不上，关掉更快也更省。思考模式下 temperature 不起作用。
        """
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "enabled" if thinking else "disabled"},
            "max_tokens": max_tokens,
        }
        if not thinking:
            body["temperature"] = 0         # 尽量让同样的输入得到同样的输出
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        url = f"{self.base_url}/chat/completions"

        for attempt in range(retries + 1):
            try:
                resp = httpx.post(url, json=body, headers=headers, timeout=self.timeout)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                problem = f"网络错误：{type(exc).__name__}"          # 网络问题，可以重试
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    usage = data.get("usage") or {}
                    self.prompt_tokens += usage.get("prompt_tokens", 0)
                    self.completion_tokens += usage.get("completion_tokens", 0)
                    self.calls += 1
                    message = data["choices"][0]["message"]
                    # 思考模式下，思考过程放在 reasoning_content 里，和最终回答 content 分开
                    self.last_reasoning = message.get("reasoning_content")
                    content = (message["content"] or "").strip()
                    if content or not json_mode:
                        return content
                    # DeepSeek 文档提到：JSON 模式偶尔会返回空内容，重试一下
                    problem = "返回内容为空"
                elif resp.status_code not in RETRYABLE_STATUS:
                    # 比如 401（key 不对）、400（参数不对），重试也没用，直接报错
                    raise LLMError(f"HTTP {resp.status_code}：{resp.text[:200]}")
                else:
                    problem = f"HTTP {resp.status_code}"
            if attempt < retries:
                time.sleep(2 ** attempt)    # 等 1 秒、2 秒……再试
        raise LLMError(f"重试 {retries} 次后仍然失败（{problem}）")

    def chat_json(self, prompt: str, *, max_tokens: int = 800, retries: int = 1,
                  thinking: bool = False) -> dict:
        """用 JSON 模式提问，返回解析好的 dict。解析不了就重试，还不行就抛 LLMError。"""
        for attempt in range(retries + 1):
            text = self.chat(prompt, max_tokens=max_tokens, json_mode=True, thinking=thinking)
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                problem = f"返回的不是合法 JSON：{text[:100]}"
                continue
            if isinstance(data, dict):
                return data
            problem = "返回的 JSON 不是一个对象"
        raise LLMError(problem)

    def usage_report(self) -> str:
        return (f"共调用 {self.calls} 次，输入 {self.prompt_tokens} token，"
                f"输出 {self.completion_tokens} token")
