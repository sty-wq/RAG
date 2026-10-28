"""问答服务（FastAPI）。

启动：
  .venv/bin/uvicorn api:app --reload
然后打开 http://127.0.0.1:8000/docs ，可以直接在网页上试接口。

接口：
  GET  /health   服务是否正常、向量库是用哪个模型建的
  POST /ask      {"question": "……", "k": 5} → 回答 + 出处

注意：Milvus Lite 同一时间只允许一个进程打开数据库文件，
所以服务运行时不能同时跑 ingest.py；要重新建库，先停掉服务。
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from answer import AnswerError, answer_question


@dataclass
class Deps:
    """服务要用到的三样东西。测试时传入假的，正式运行时在启动时创建真的。"""
    embedder: object
    store: object
    llm: object


def build_real_deps() -> Deps:
    from embedder import Embedder
    from llm import DeepSeekClient
    from vector_store import VectorStore
    return Deps(embedder=Embedder(), store=VectorStore(), llm=DeepSeekClient())


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)   # 先去掉首尾空白再检查，只有空格的问题会被拒绝

    question: str = Field(min_length=1, max_length=2000, description="你的问题")
    k: int = Field(default=5, ge=1, le=20, description="最多参考几块资料")


class SourceOut(BaseModel):
    index: int
    doc_name: str
    heading_path: str
    source_path: str
    score: float
    text: str
    cited: bool


class AskResponse(BaseModel):
    question: str
    answer: str
    found: bool
    sources: list[SourceOut]


def create_app(deps: Deps | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 服务启动时准备好依赖，关闭时释放（Milvus Lite 的文件锁要记得放开）
        app.state.deps = deps or build_real_deps()
        yield
        if deps is None:
            app.state.deps.store.close()

    app = FastAPI(title="RAG 笔记问答", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        meta = app.state.deps.store.read_meta()
        return {"status": "ok", "index": meta}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        d = app.state.deps
        try:
            result = answer_question(req.question, d.embedder, d.store, d.llm, k=req.k)
        except AnswerError as exc:
            # 还没建库、embedding 模型对不上：服务暂时不可用
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            # 调用 DeepSeek 或 embedding 失败：上游服务出错
            raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc
        return AskResponse(**asdict(result))

    return app


app = create_app()
