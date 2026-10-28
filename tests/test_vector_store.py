"""Milvus Lite 入库和检索：用真的 Milvus Lite（本地文件），向量是手写的小向量。

没装 pymilvus 时自动跳过；CI 会安装它，所以在 CI 里一定会跑。
"""
from __future__ import annotations

import pytest

pytest.importorskip("pymilvus")
pytest.importorskip("milvus_lite")

from chunks import Chunk  # noqa: E402
from vector_store import VectorStore, fit_bytes  # noqa: E402


def make_chunk(i: int, text: str) -> Chunk:
    return Chunk(id=f"chunk-{i}", doc_id="doc-1", doc_name="笔记.md", source_path="notes/笔记.md",
                 section_id=i, heading_path=f"笔记.md > 第{i}节", order=i, text=text, embed_text=text)


CHUNKS = [make_chunk(0, "苹果"), make_chunk(1, "香蕉"), make_chunk(2, "汽车")]
VECTORS = [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]


def test_fit_bytes_never_cuts_a_character_in_half():
    text, cut = fit_bytes("汉字汉字", 7)          # 一个汉字 3 个字节，7 个字节只放得下 2 个
    assert text == "汉字" and cut
    assert fit_bytes("abc", 10) == ("abc", False)


def test_rebuild_then_search_returns_nearest_first(tmp_path):
    with VectorStore(uri=str(tmp_path / "data" / "test.db")) as store:
        assert store.read_meta() is None                          # 还没建库
        store.rebuild(CHUNKS, VECTORS, embed_model="fake-model")
        hits = store.search([1.0, 0.05, 0.0, 0.0], k=2)
        meta = store.read_meta()

    assert [h.text for h in hits] == ["苹果", "香蕉"]
    assert hits[0].score > hits[1].score
    assert hits[0].heading_path == "笔记.md > 第0节" and hits[0].source_path == "notes/笔记.md"
    assert meta["embed_model"] == "fake-model" and meta["dim"] == 4 and meta["chunk_count"] == 3


def test_search_works_after_collection_is_released(tmp_path):
    """回归测试：单独运行 search.py 时，集合在磁盘上但没加载进内存（released），检索要自己先加载。"""
    with VectorStore(uri=str(tmp_path / "test.db")) as store:
        store.rebuild(CHUNKS, VECTORS, embed_model="fake-model")
        store.client.release_collection(store.collection)        # 模拟"换了一个进程"
        hits = store.search([0.0, 0.0, 1.0, 0.0], k=1)
    assert hits[0].text == "汽车"


def test_rebuild_replaces_old_data(tmp_path):
    uri = str(tmp_path / "test.db")
    with VectorStore(uri=uri) as store:
        store.rebuild(CHUNKS, VECTORS, embed_model="fake-model")
        store.rebuild(CHUNKS[:1], VECTORS[:1], embed_model="fake-model")
        hits = store.search([0.0, 0.0, 1.0, 0.0], k=5)
    assert [h.text for h in hits] == ["苹果"]                     # 旧的三块已经没有了


def test_very_long_text_is_truncated_not_rejected(tmp_path):
    long_chunk = make_chunk(0, "长" * 30000)                      # 9 万字节，超过 65535
    with VectorStore(uri=str(tmp_path / "test.db")) as store:
        truncated = store.rebuild([long_chunk], [VECTORS[0]], embed_model="fake-model")
        hits = store.search(VECTORS[0], k=1)
    assert truncated == 1
    assert len(hits[0].text.encode("utf-8")) <= 65535
