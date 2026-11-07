"""共享检索流程：dense / BM25 / RRF 混合召回 → 可选模型重排 → 最终 top-k。

API、命令行和评测共用这里。RRF 只融合名次，不把 BM25 和余弦分数相加。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os

from vector_store import SearchHit


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str = "dense"
    candidate_k: int = 50         # 每路召回数，也是 RRF 融合后送去重排的候选池上限
    rrf_k: int = 60               # RRF 平滑常数，和返回几条不是一回事
    min_rerank_score: float | None = None

    def __post_init__(self):
        if self.mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError("mode 必须是 dense、bm25 或 hybrid")
        if isinstance(self.candidate_k, bool) or not isinstance(self.candidate_k, int) or not 1 <= self.candidate_k <= 1000:
            raise ValueError("candidate_k 必须是 1~1000 的整数")
        if isinstance(self.rrf_k, bool) or not isinstance(self.rrf_k, int) or self.rrf_k < 1:
            raise ValueError("rrf_k 必须是正整数")
        if self.min_rerank_score is not None and not math.isfinite(self.min_rerank_score):
            raise ValueError("min_rerank_score 必须是有限数值")

    @classmethod
    def from_env(cls, default_mode: str = "hybrid") -> RetrievalConfig:
        threshold = os.getenv("RAG_MIN_RERANK_SCORE")
        return cls(mode=os.getenv("RAG_RETRIEVAL_MODE", default_mode),
                   candidate_k=int(os.getenv("RAG_CANDIDATE_K", "50")),
                   rrf_k=int(os.getenv("RAG_RRF_K", "60")),
                   min_rerank_score=float(threshold) if threshold else None)


def fuse_rrf(dense: list[SearchHit], lexical: list[SearchHit],
             k: int, rrf_k: int = 60) -> list[SearchHit]:
    """按 chunk_id 去重；保留各路原始分数，避免丢掉仅被 BM25 找到的块。"""
    if k < 1 or rrf_k < 1:
        raise ValueError("k 和 rrf_k 必须大于 0")
    hits: dict[str, SearchHit] = {}
    scores: dict[str, float] = {}
    for route, ranking in (("dense_score", dense), ("lexical_score", lexical)):
        seen = set()
        for rank, hit in enumerate(ranking, 1):
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            if hit.chunk_id not in hits:
                hits[hit.chunk_id] = replace(hit, dense_score=None, lexical_score=None,
                                             rrf_score=None, rerank_score=None)
            hits[hit.chunk_id] = replace(hits[hit.chunk_id], **{route: hit.score})
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    # 相同融合分数按 chunk_id 排序，跨进程也稳定。
    ids = sorted(hits, key=lambda key: (-scores[key], key))[:k]
    return [replace(hits[key], score=scores[key], score_type="rrf", rrf_score=scores[key]) for key in ids]


def rerank_hits(question: str, hits: list[SearchHit], reranker, k: int) -> list[SearchHit]:
    from reranker import RerankError

    if not hits:
        return []
    documents = [f"{hit.heading_path}\n{hit.text}" for hit in hits]
    scores = reranker.score(question, documents)
    try:
        scores = [float(score) for score in scores]
    except (TypeError, ValueError, OverflowError) as exc:
        raise RerankError("重排模型返回了非数值分数") from exc
    if len(scores) != len(hits) or not all(math.isfinite(score) for score in scores):
        raise RerankError("重排分数必须有限，且与候选逐一对应")
    ranking = sorted(range(len(hits)), key=lambda index: (-scores[index], index))[:k]
    score_type = "rerank_rank" if getattr(reranker, "rank_only", False) else "rerank"
    return [replace(hits[index], score=scores[index], score_type=score_type, rerank_score=scores[index],
                    dense_score=(hits[index].score if hits[index].score_type == "cosine"
                                 else hits[index].dense_score)) for index in ranking]


def retrieve(question: str, query_vector: list[float] | None, store, k: int = 5, *,
             config: RetrievalConfig | None = None, reranker=None,
             doc_name: str | None = None, rerank_question: str | None = None) -> list[SearchHit]:
    """question 用于召回；显式的 rerank_question 仅替换重排问题，不改候选文字。"""
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 1000:
        raise ValueError("k 必须是 1~1000 的整数")
    if rerank_question is not None and (not isinstance(rerank_question, str) or not rerank_question.strip()):
        raise ValueError("rerank_question 必须是非空字符串或 None")
    config = config or RetrievalConfig()
    if getattr(reranker, "rank_only", False) and config.min_rerank_score is not None:
        raise ValueError("DeepSeek 返回名次而非相关性分数，不支持 --min-rerank-score")
    pool_k = max(k, config.candidate_k) if config.mode == "hybrid" or reranker is not None else k
    filters = {"doc_name": doc_name} if doc_name is not None else {}
    dense = store.search(query_vector, pool_k, **filters) if config.mode != "bm25" else []
    lexical = store.lexical_search(question, pool_k, **filters) if config.mode != "dense" else []
    if config.mode == "hybrid":
        hits = fuse_rrf(dense, lexical, pool_k, config.rrf_k)
    else:
        hits = dense if config.mode == "dense" else lexical
    scoring_question = question if rerank_question is None else rerank_question
    return rerank_hits(scoring_question, hits, reranker, k) if reranker is not None else hits[:k]


def passes_score_gate(hit, min_dense_score: float, config: RetrievalConfig | None = None) -> bool:
    """0.35 只用于余弦分数；BM25 / RRF 没有可复用的绝对门槛。"""
    kind = getattr(hit, "score_type", "cosine")
    if kind == "rerank_rank":
        if config is not None and config.min_rerank_score is not None:
            raise ValueError("名次不能用作拒答门槛")
        return True
    if kind == "cosine":
        return hit.score >= min_dense_score
    if kind == "rerank" and config is not None and config.min_rerank_score is not None:
        return hit.score >= config.min_rerank_score
    return True


def add_retrieval_args(parser, *, default_rerank: str = "none",
                       defaults: RetrievalConfig | None = None) -> None:
    defaults = defaults or RetrievalConfig()
    parser.add_argument("--mode", choices=("dense", "bm25", "hybrid"), default=defaults.mode)
    parser.add_argument("--candidate-k", type=int, default=defaults.candidate_k, help="每路召回和融合候选池大小，默认 50")
    parser.add_argument("--rrf-k", type=int, default=defaults.rrf_k, help="RRF 平滑常数，默认 60")
    parser.add_argument("--rerank", nargs="?", const="local", choices=("none", "local", "http", "deepseek"),
                        default=default_rerank,
                        help="不带值等于 local；http 调用 /rerank；deepseek 使用 .env.ocr 的聊天模型排序（计费）")
    parser.add_argument("--min-rerank-score", type=float, default=defaults.min_rerank_score,
                        help="可选重排门槛，必须针对所选模型校准；默认不按重排分数拒答")


def config_from_args(args) -> RetrievalConfig:
    return RetrievalConfig(args.mode, args.candidate_k, args.rrf_k, args.min_rerank_score)
