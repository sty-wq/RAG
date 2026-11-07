"""Only query embeddings and Dense retrieval; no ingestion, reranker, or LLM.

Uses the frozen index copy in this experiment directory. Resume preserves each
query vector and both k=10 / k=50 searches. Does not touch the production index.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from dataclasses import asdict

OUT = Path(__file__).resolve().parent
ROOT = Path('/Users/bia/PycharmProjects/PythonProject5')
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False))
    temp.replace(p)


def main():
    manifest = json.loads((OUT/'manifest.json').read_text())
    for name, digest in manifest['project_code'].items():
        if sha(ROOT/name) != digest:
            raise SystemExit(f'代码已变化，停止复用冻结配置：{name}')
    for name, digest in manifest['frozen_files'].items():
        if sha(OUT/name) != digest:
            raise SystemExit(f'冻结文件不一致：{name}')
    qs = [json.loads(s) for s in (OUT/'frozen/questions_ctx.jsonl').read_text().splitlines()]
    qs = [q for q in qs if q['should_answer'] and any(q['evidence'])]
    assert len(qs) == 1309
    with (OUT/'.dense.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pending = [q for q in qs if not (OUT/'dense1309'/f"{q['qid']}.json").exists()]
        if not pending:
            print('Dense 缓存已完整：1309/1309', flush=True)
            return
        if not os.getenv('EMBED_API_KEY'):
            raise SystemExit('缺少 EMBED_API_KEY。请在此前能运行 embedding 的终端执行，不要发送密钥到聊天。')
        from embedder import Embedder
        from vector_store import VectorStore
        embedder = Embedder(model='text-embedding-3-small', base_url='https://api.quickrouter.ai/v1')
        with VectorStore(uri=str(OUT/'index-working/milvus.db')) as store:
            assert int(store.client.get_collection_stats(store.collection)['row_count']) == 17887
            for start in range(0, len(pending), 10):
                batch = pending[start:start+10]
                missing = [q for q in batch if not (OUT/'vectors1309'/f"{q['qid']}.json").exists()]
                if missing:
                    vectors = embedder.embed([q['question'] for q in missing])
                    for q, v in zip(missing, vectors, strict=True):
                        assert len(v) == 1536
                        write(OUT/'vectors1309'/f"{q['qid']}.json", {'question':q['question'],'vector':v})
                for q in batch:
                    v = json.loads((OUT/'vectors1309'/f"{q['qid']}.json").read_text())
                    assert v['question'] == q['question']
                    hits50 = store.search(v['vector'], 50)
                    hits10 = store.search(v['vector'], 10)
                    assert len(hits50) == len({h.chunk_id for h in hits50}) == 50
                    write(OUT/'dense1309'/f"{q['qid']}.json", {
                        'qid':q['qid'], 'question':q['question'],
                        'hits50':[asdict(h) for h in hits50], 'hits10':[asdict(h) for h in hits10]})
                print(f'Dense 候选已保存 {len(qs)-len(pending)+start+len(batch)}/{len(qs)}',flush=True)


if __name__ == '__main__':
    main()
