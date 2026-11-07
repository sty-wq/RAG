"""真实 Milvus Lite：旧 schema 可直接构建 BM25，过滤与缓存不能返回旧集合。"""
import pytest

pytest.importorskip("pymilvus")
pytest.importorskip("milvus_lite")

from chunks import Chunk  # noqa: E402
from retrieval import RetrievalConfig, retrieve  # noqa: E402
from vector_store import VectorStore  # noqa: E402


def chunk(index, text, doc="a.md"):
    return Chunk(f"chunk-{index}", f"doc-{doc}", doc, f"notes/{doc}", index,
                 f"{doc} > 章节", index, text, text)


def test_bm25_loads_existing_collection_filters_and_survives_reopen(tmp_path):
    uri = str(tmp_path / "test.db")
    chunks = [chunk(0, "红黑树实现 requestId 幂等"), chunk(1, "苹果香蕉", 'other".md')]
    with VectorStore(uri) as store:
        store.rebuild(chunks, [[1.0, 0, 0, 0], [0.0, 1, 0, 0]], "m")
        # rebuild 仍用原 schema，不增加字段，也不要求重新生成已有 embedding。
    with VectorStore(uri) as store:
        assert store.lexical_search("红黑树")[0].text == chunks[0].text
        assert store.lexical_search("苹果", doc_name='other".md')[0].chunk_id == "chunk-1"
        assert store.lexical_search("苹果", doc_name="a.md") == []
        assert store.lexical_search("苹果", doc_name="missing.md") == []
        assert store.lexical_search("香蕉")[0].score_type == "bm25"


def test_rebuild_invalidates_cached_bm25_and_no_deleted_text_leaks(tmp_path):
    with VectorStore(str(tmp_path / "test.db")) as store:
        store.rebuild([chunk(0, "苹果")], [[1.0, 0, 0, 0]], "m")
        assert store.lexical_search("苹果")
        store.rebuild([chunk(1, "香蕉")], [[1.0, 0, 0, 0]], "m")
        assert store.lexical_search("苹果") == []
        assert store.lexical_search("香蕉")[0].chunk_id == "chunk-1"


def test_bm25_iterator_does_not_truncate_corpus_at_one_batch(tmp_path):
    chunks = [chunk(i, f"filler {i}") for i in range(1001)] + [chunk(1001, "UNIQUE_TAIL_TOKEN")]
    with VectorStore(str(tmp_path / "test.db")) as store:
        store.rebuild(chunks, [[1.0, 0, 0, 0] for _ in chunks], "m")
        hits = store.lexical_search("UNIQUE_TAIL_TOKEN", 3)
        assert [h.chunk_id for h in hits] == ["chunk-1001"]


def test_hybrid_adds_exact_keyword_match_missing_from_dense_candidates(tmp_path):
    chunks = [chunk(0, "数据库的一般介绍"), chunk(1, "错误 ERR_4097 的解决办法")]
    with VectorStore(str(tmp_path / "test.db")) as store:
        store.rebuild(chunks, [[1.0, 0, 0, 0], [0.0, 1, 0, 0]], "m")
        dense = store.search([1.0, 0, 0, 0], 1)
        assert dense[0].chunk_id == "chunk-0"
        hybrid = retrieve("ERR_4097", [1.0, 0, 0, 0], store, 2, config=RetrievalConfig("hybrid", 1))
        assert {h.chunk_id for h in hybrid} == {"chunk-0", "chunk-1"}
        assert hybrid[0].chunk_id == "chunk-1"
        keyword = next(h for h in hybrid if h.chunk_id == "chunk-1")
        assert keyword.lexical_score > 0 and keyword.score_type == "rrf"
