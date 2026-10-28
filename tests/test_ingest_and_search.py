"""建库和检索的整体流程：embedding 和向量库都换成假的，只测流程本身。"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

from ingest import run_ingest
from search import run_search


class FakeEmbedder:
    model = "fake-embed"

    def embed(self, texts):
        # 用文字的哈希造一个 3 维"向量"，同样的文字得到同样的向量
        return [[b / 255 for b in hashlib.md5(t.encode()).digest()[:3]] for t in texts]

    def embed_one(self, text):
        return self.embed([text])[0]


class FakeStore:
    def __init__(self, meta=None):
        self.meta = meta
        self.rebuilt = None
        self.searched = None

    def rebuild(self, chunks, vectors, embed_model):
        self.rebuilt = (chunks, vectors, embed_model)
        self.meta = {"embed_model": embed_model}
        return 0

    def read_meta(self):
        return self.meta

    def search(self, vector, k):
        self.searched = (vector, k)
        return [SimpleNamespace(score=0.9, heading_path="a.md > 标题", text="内容")]


def test_ingest_without_llm(tmp_path):
    (tmp_path / "a.md").write_text("# 标题\n内容一\n## 子标题\n内容二", encoding="utf-8")
    (tmp_path / "empty.md").write_text("   ", encoding="utf-8")
    (tmp_path / "x.csv").write_text("1,2", encoding="utf-8")
    store = FakeStore()

    report = run_ingest(str(tmp_path), FakeEmbedder(), store)

    chunks, vectors, model = store.rebuilt
    assert report["documents"] == 1 and report["chunks"] == 2 and report["dim"] == 3
    assert report["empty"] == ["empty.md"] and len(report["failed"]) == 1
    assert model == "fake-embed" and len(vectors) == len(chunks)
    assert chunks[1].heading_path == "a.md > 标题 > 子标题"


def test_ingest_with_llm_applies_structure_fixes(tmp_path, fake_llm, monkeypatch):
    import outline
    import structure_fix
    monkeypatch.setattr(outline, "SUMMARY_CACHE_FILE", tmp_path / "s.json")
    monkeypatch.setattr(structure_fix, "FIX_CACHE_FILE", tmp_path / "f.json")
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "q.md").write_text("# 题目4\n题面\n# 参考代码\n代码", encoding="utf-8")
    llm = fake_llm({"fixes": [{"section_id": 2, "new_level": 2, "reason": "是题目4的代码"}]})
    store = FakeStore()

    report = run_ingest(str(tmp_path / "notes"), FakeEmbedder(), store, llm=llm)

    assert report["structure_fixes"] == 1
    assert store.rebuilt[0][1].heading_path == "q.md > 题目4 > 参考代码"


def test_nothing_to_ingest(tmp_path):
    store = FakeStore()
    report = run_ingest(str(tmp_path), FakeEmbedder(), store)
    assert report["chunks"] == 0 and store.rebuilt is None


def test_search_refuses_when_not_built():
    hits, error = run_search("问题", FakeEmbedder(), FakeStore(meta=None))
    assert hits == [] and "ingest" in error


def test_search_refuses_mixed_models():
    hits, error = run_search("问题", FakeEmbedder(), FakeStore(meta={"embed_model": "另一个模型"}))
    assert hits == [] and "另一个模型" in error


def test_search_returns_hits():
    store = FakeStore(meta={"embed_model": "fake-embed"})
    hits, error = run_search("问题", FakeEmbedder(), store, k=3)
    assert error is None and hits[0].text == "内容"
    assert store.searched[1] == 3
