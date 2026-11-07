import fcntl
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
import sys
OUT=Path(__file__).resolve().parent
BENCH=OUT.parent
ROOT=Path('/Users/bia/PycharmProjects/PythonProject5')
sys.path.insert(0,str(ROOT))
from embedder import Embedder
from vector_store import VectorStore

def read(p): return list(map(json.loads,p.read_text().splitlines()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False));t.replace(p)
def cached(p,fn):
    if not p.exists():write(p,fn())
    return json.loads(p.read_text())
QFILE=BENCH/'fulltext-screen/questions_correct.jsonl'
qs=read(QFILE)
assert len(qs)==len({q['qid'] for q in qs})==93 and sha(QFILE)==json.loads((QFILE.parent/'summary.json').read_text())['retained_sha256']
CORPUS=ROOT/'evaluation/data/qasper/corpus'
def prepare():
    with (OUT/'.pool.lock').open('a') as lk:
        fcntl.flock(lk,fcntl.LOCK_EX)
        db=CORPUS.parent/'index/milvus.db'; meta=Path(str(db)+'.meta.json')
        settings=dict(dataset=sha(QFILE),index=sha(meta),model='text-embedding-3-small',base_url='https://api.quickrouter.ai/v1',k=50)
        if cached(OUT/'pool-config.json',lambda:settings)!=settings:raise ValueError('Dense 缓存版本不一致')
        pending=[q for q in qs if not (OUT/'items'/q['qid']/'pool.json').exists()]
        if not pending:return
        if not os.getenv('EMBED_API_KEY'):raise SystemExit('生成查询向量需要 EMBED_API_KEY；请在已配置该变量的终端执行脚本。')
        assert json.loads(meta.read_text())['embed_model']==settings['model']
        embedder=Embedder(model=settings['model'],base_url=settings['base_url'])
        with VectorStore(uri=str(db)) as store:
            for start in range(0,len(pending),10):
                batch=pending[start:start+10]; vectors=embedder.embed([q['question'] for q in batch])
                for q,v in zip(batch,vectors,strict=True):
                    hits=store.search(v,50)
                    assert len({h.chunk_id for h in hits})==len(hits)==50
                    d=OUT/'items'/q['qid'];d.mkdir(parents=True,exist_ok=True)
                    write(d/'pool.json',[asdict(h) for h in hits])
                print('Dense 候选已保存',len(qs)-len(pending)+start+len(batch),'/',len(qs),flush=True)
if __name__=='__main__':prepare()
