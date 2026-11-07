"""Verify replay against production fusion/BM25 implementations, without APIs."""
import importlib.util
import json
from pathlib import Path
import sys
import threading

OUT=Path(__file__).resolve().parent
ROOT=Path('/Users/bia/PycharmProjects/PythonProject5')
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
from vector_store import SearchHit, VectorStore
from retrieval import fuse_rrf

spec=importlib.util.spec_from_file_location('ablation',OUT/'analyze.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
rows=a.chunks();lookup={r['id']:r for r in rows}
def hit(cid,score=0):
    h=lookup[cid]
    return SearchHit(score=score,chunk_id=cid,**{k:h[k] for k in
               ('doc_name','source_path','heading_path','chunk_order','text')})

checked=0
for folder in ('rewritten500','original1309'):
    path=OUT/folder/'details.jsonl'
    if not path.exists():continue
    for r in a.read(path):
        d=r['rankings']['dense'];b=r['rankings']['bm25']
        expected=[h.chunk_id for h in fuse_rrf([hit(c) for c in d],[hit(c) for c in b],50,60)]
        assert expected==r['rankings']['rrf_a1']
        assert a.weighted(d,b,0)==d
        checked+=1

# Storage-independent parity: exercise the actual VectorStore.lexical_search
# with a fake iterator deliberately in reverse ID order, then compare top50.
class Iterator:
    def __init__(self):self.left=list(reversed(rows))
    def next(self):
        result,self.left=self.left[:500],self.left[500:]
        return result
    def close(self):pass
class Client:
    def load_collection(self,*args):pass
    def query_iterator(self,*args,**kwargs):return Iterator()
store=VectorStore.__new__(VectorStore)
store.uri=str(OUT/'nonexistent.db');store.collection='notes'
store.client=Client();store._lexical_cache=None;store._lexical_lock=threading.Lock()
qs=a.read(OUT/'frozen/questions_ctx.jsonl')
lex={r['qid']:r['hits'] for r in a.read(OUT/'bm25-original.jsonl')}
for q in [qs[i] for i in (0,1,10,100,500,1000,1300,1450)]:
    actual=store.lexical_search(q['question'],50)
    expected=lex[q['qid']]
    assert [h.chunk_id for h in actual]==[h['chunk_id'] for h in expected]
    assert [h.score for h in actual]==[h['score'] for h in expected]

# Boundary invariant including RRF ties; absent route contributes zero.
d=['dense-best']+[f'd{i}' for i in range(48)]+['shared-worst']
b=['bm-best']+[f'b{i}' for i in range(48)]+['shared-worst']
assert a.weighted(d,b,1)[0]=='shared-worst'
assert a.weighted(d,b,0)==d
assert a.weighted(['z','a'],['a','z'],1)==['a','z']
assert a.weighted(['z','a'],['a','z'],0)==['z','a']
assert a.weighted(['d'],['b'],.25)==['d','b']
result={'real_candidate_fusions_match_production':checked,
        'alpha_zero_exact_dense_order':checked,
        'bm25_top50_matches_production_fake_iterator_queries':8,
        'boundary_and_tie_assertions':5,'network_calls':0,
        'analysis_sha256':a.sha(OUT/'analyze.py'),'verification_sha256':a.sha(Path(__file__))}
a.write(OUT/'verification.json',result)
print(json.dumps(result,ensure_ascii=False))
