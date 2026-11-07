"""Freeze input artifacts once. No API calls or production writes."""
import hashlib
import json
from pathlib import Path
import shutil

OUT = Path(__file__).resolve().parent
ROOT = Path('/Users/bia/PycharmProjects/PythonProject5')
DATA = ROOT/'evaluation/data/qasper'
BASE = OUT.parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not (OUT/'manifest.json').exists(), 'Already frozen'
    frozen = OUT/'frozen'
    frozen.mkdir(exist_ok=True)
    sources = {
        'questions_ctx.jsonl': DATA/'questions_ctx.jsonl',
        'questions500.jsonl': BASE/'qasper-500/questions_ready_500.jsonl',
        'sc500-run-config.json': BASE/'sc500-answer-review/run_config.json',
        'index-meta.json': DATA/'index/milvus.db.meta.json',
    }
    for tag, stamp in [('A','20261102-171638'),('B','20261102-172040'),('C','20261102-172530')]:
        sources[f'history-{tag}.jsonl'] = DATA/f'runs/{stamp}.jsonl'
        sources[f'history-{tag}-summary.json'] = next((ROOT/'evaluation/results').glob(f'{stamp}_*.json'))
    names = ['retrieval.py','lexical.py','vector_store.py','embedder.py','chunks.py',
             'chunker.py','main.py','ocr_service.py','evaluation/metrics.py']
    for name in names:
        sources['code/'+name] = ROOT/name
    for name, src in sources.items():
        dest = frozen/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    # Immutable source index snapshot; Milvus runtime uses a separate copy.
    shutil.copytree(DATA/'index/milvus.db', frozen/'milvus.db')
    (OUT/'index-working').mkdir(exist_ok=True)
    shutil.copytree(frozen/'milvus.db', OUT/'index-working/milvus.db')
    shutil.copy2(frozen/'index-meta.json', OUT/'index-working/milvus.db.meta.json')
    files = {str(p.relative_to(OUT)):sha(p) for p in sorted(frozen.rglob('*')) if p.is_file()}
    corpus = {p.name:sha(p) for p in sorted((DATA/'corpus').glob('*.md'))}
    source_manifest = json.loads((frozen/'milvus.db/collections/notes/manifest.json').read_text())
    assert len(source_manifest['partitions']['_default']['data_files']) == 1
    assert not source_manifest['partitions']['_default']['delta_files']
    qs = [json.loads(s) for s in (frozen/'questions500.jsonl').read_text().splitlines()]
    cfg = json.loads((frozen/'sc500-run-config.json').read_text())
    assert cfg['questions'] == sha(frozen/'questions500.jsonl')
    assert cfg['index_meta'] == sha(frozen/'index-meta.json')
    pools = OUT/'dense500'
    pools.mkdir()
    for q in qs:
        src = BASE/'sc500-answer-review/items'/q['qid']/'pool.json'
        hits = json.loads(src.read_text())
        assert len(hits) == len({h['chunk_id'] for h in hits}) == 50
        dest = pools/f"{q['qid']}.json"
        shutil.copy2(src, dest)
        files[str(dest.relative_to(OUT))] = sha(dest)
    manifest = {
        'experiment':'hybrid-ablation-p0-p2', 'created_date':'2026-11-04',
        'primary':'original 1309 answerable questions_ctx',
        'secondary':'500 rewritten questions, existing Dense50 replay; not a held-out test',
        'embedding':{'model':'text-embedding-3-small','endpoint':'https://api.quickrouter.ai/v1'},
        'candidate_k':50,'rrf_k':60,'alphas':[0,.25,.5,1],
        'project_code':{name:sha(ROOT/name) for name in names},
        'sources':{name:str(src) for name,src in sources.items()},
        'corpus_sha256':corpus,'frozen_files':files,
        'notes':['No new chunking, embeddings for documents, reranking or answer generation.',
                 'Primary query embeddings may require new API calls; save for replay.',
                 'Union up to 100 is an upper bound, not a same-budget competitor.',
                 'Historical input hashes were not frozen then; top10 identity and metrics must be checked.'],
    }
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('Frozen',len(corpus),'papers,',len(qs),'cached Dense50 pools;',len(files),'hashed files')


if __name__ == '__main__':
    main()
