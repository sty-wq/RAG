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
from dataclasses import asdict, dataclass, replace
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from answer import CandidateInputError, AnswerError, answer_question
from evidence_coverage import Coverage, assess_coverage
from retrieval import RetrievalConfig


@dataclass
class Deps:
    """服务要用到的三样东西。测试时传入假的，正式运行时在启动时创建真的。"""
    embedder: object
    store: object
    llm: object
    retrieval_config: RetrievalConfig | None = None
    reranker: object | None = None


def build_real_deps() -> Deps:
    from embedder import Embedder
    from llm import DeepSeekClient
    from vector_store import VectorStore
    from reranker import create_reranker
    # 配置先检查，最后打开数据库，避免配置错误时留下未关闭的文件锁。
    config = RetrievalConfig.from_env()
    reranker = create_reranker()
    embedder, llm = Embedder(), DeepSeekClient()
    return Deps(embedder=embedder, store=VectorStore(), llm=llm,
                retrieval_config=config, reranker=reranker)


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)   # 先去掉首尾空白再检查，只有空格的问题会被拒绝

    question: str = Field(min_length=1, max_length=2000, description="你的问题")
    k: int = Field(default=5, ge=1, le=20, description="最多参考几块资料")
    mode: Literal["dense", "bm25", "hybrid"] | None = Field(default=None, description="检索方式，默认采用服务配置")
    candidate_k: int | None = Field(default=None, ge=1, le=1000, description="召回和重排候选池大小")
    evidence_coverage: bool = Field(default=False, description="核验必答要点与逐项引用")
    one_pass: bool = Field(default=False, description="Dense 候选直接交给 LLM 作答，候选数由 candidate_k 控制")
    rerank: bool | None = Field(default=None, description="是否启用服务已配置的重排模型")


class SourceOut(BaseModel):
    index: int
    doc_name: str
    heading_path: str
    source_path: str
    score: float
    text: str
    cited: bool
    score_type: str = "cosine"
    dense_score: float | None = None
    lexical_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    found: bool
    sources: list[SourceOut]
    evidence_coverage: Coverage | None = None


def create_app(deps: Deps | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 服务启动时准备好依赖，关闭时释放（Milvus Lite 的文件锁要记得放开）
        app.state.deps = deps or build_real_deps()
        try:
            yield
        finally:
            if deps is None:
                try:
                    app.state.deps.store.close()
                finally:
                    close = getattr(app.state.deps.reranker, "close", None)
                    if close is not None:
                        close()

    app = FastAPI(title="RAG 笔记问答", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        d = app.state.deps
        meta = d.store.read_meta()
        return {"status": "ok", "index": meta, "retrieval": asdict(d.retrieval_config or RetrievalConfig()),
                "reranker": d.reranker.describe() if d.reranker is not None else None}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        d = app.state.deps
        config = d.retrieval_config or RetrievalConfig()
        if req.one_pass and (req.rerank is True or req.mode not in (None, "dense")):
            raise HTTPException(status_code=422, detail="one_pass 只支持 dense 且不能启用独立重排")
        mode = "dense" if req.one_pass else req.mode or config.mode
        config = replace(config, mode=mode,
                         candidate_k=req.candidate_k if req.candidate_k is not None else config.candidate_k)
        reranker = d.reranker if not req.one_pass and req.rerank is not False else None
        if req.rerank is True and reranker is None:
            raise HTTPException(status_code=503, detail="服务未配置重排模型，请设置 RERANK_BACKEND=local 或 http")
        try:
            result = answer_question(req.question, d.embedder, d.store, d.llm, k=req.k,
                                     config=config, reranker=reranker, one_pass=req.one_pass)
        except CandidateInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except AnswerError as exc:
            # 还没建库、embedding 模型对不上：服务暂时不可用
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            # 调用 DeepSeek 或 embedding 失败：上游服务出错
            raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc
        coverage = assess_coverage(req.question, result.answer, result.sources, d.llm) if req.evidence_coverage else None
        return AskResponse(**asdict(result), evidence_coverage=coverage)

    return app


app = create_app()
