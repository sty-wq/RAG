"""Read-only metric alignment from saved experiments. No model/API calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import mean

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
ROOT=Path('/Users/bia/PycharmProjects/PythonProject5')
spec=importlib.util.spec_from_file_location('metrics',ROOT/'evaluation/metrics.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
hashes={}

def read(p):
    hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())

def rows(p):
    hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]

def rank_metrics(cov,groups):
    out={}
    for k in (1,5,10,50):
        if len(cov)<k:continue
        out.update({f'hit@{k}':m.hit_at_k_covered(cov,k),
                    f'recall@{k}':m.recall_at_k_covered(cov,groups,k),
                    f'all_evidence@{k}':int(m.recall_at_k_covered(cov,groups,k)==1)})
    for k in (5,10):
        if len(cov)>=k:out[f'mrr@{k}']=m.reciprocal_rank_covered(cov[:k])
    return out

def agg(xs):
    assert xs
    return {k:mean(r[k] for r in xs) for k in xs[0]}

def main():
    single=BASE/'sc500-answer-review'
    qs=rows(BASE/'qasper-500/questions_ready_500.jsonl')
    pairs={r['qid']:r for r in rows(single/'pairs.jsonl')}
    assert len(qs)==len(pairs)==500
    single_details=[]
    for q in qs:
        qid=q['qid'];r=pairs[qid]
        assert r['question']==q['question']
        pool=read(single/'items'/qid/'pool.json');top=read(single/'items'/qid/'top5.json')
        assert len(pool)==50 and len(top)==5
        assert set(h['chunk_id'] for h in top)<=set(h['chunk_id'] for h in pool)
        paras=sorted({p for g in q['evidence'] for p in g})
        groups=[{paras.index(p) for p in g} for g in q['evidence']]
        item={'qid':qid}
        for arm,hits in [('dense',pool),('rerank',top)]:
            assert [s['id'] for s in r[arm]['sources']]==[h['chunk_id'] for h in hits[:5]]
            for s,h in zip(r[arm]['sources'],hits[:5]):assert s['text']==h['text']
            cov=[{i for i,p in enumerate(paras) if m.covers(h['text'],p)} for h in hits]
            metrics=rank_metrics(cov,groups)
            assert metrics['hit@5']==r[arm]['hit5']
            correct=r[arm]['judge']['verdict']=='correct'
            refusal=r[arm]['answer'].startswith('根据现有资料无法回答这个问题。')
            item[arm]={'metrics':metrics,'correct':correct,'refusal':refusal,
                'correct_strict':correct and not refusal,
                'correct_and_supported':correct and r[arm]['judge']['citation']=='supported'}
        single_details.append(item)
    single_summary={arm:{'n':500,'retrieval':agg([r[arm]['metrics'] for r in single_details]),
        'correct':sum(r[arm]['correct'] for r in single_details),
        'correct_strict':sum(r[arm]['correct_strict'] for r in single_details),
        'correct_and_supported':sum(r[arm]['correct_and_supported'] for r in single_details),
        'answer_evidence_coverage':None,'coverage_reason':'not measured in saved500 experiment'}
        for arm in ('dense','rerank')}

    cross=BASE/'qasper-crosspaper-100/compare93'
    cqs=rows(cross.parent/'fulltext-screen/questions_correct.jsonl')
    comparisons={r['qid']:r for r in rows(cross/'comparison-details.jsonl')}
    saved_summary=read(cross/'comparison-summary.json')
    cross_details=[]
    for q in cqs:
        qid=q['qid'];n=q['required_papers'];es=q['source_evidence'];r=comparisons[qid]
        assert len(es)==len(set(q['gold_docs']))==n
        assert r['question']==q['question']
        pool=read(cross/'items'/qid/'pool.json')
        assert len(pool)==50
        paragraph_cov=[{i for i,e in enumerate(es) if h['doc_name']==e['file'] and m.covers(h['text'],e['text'])} for h in pool]
        seed=[{i for i,e in enumerate(es) if h['chunk_id'] in e['supporting_chunk_ids']} for h in pool]
        # The cross-paper benchmark marks the *required fact quote*, not just
        # any long fragment of its paragraph. This reproduces its seed IDs.
        cov=[{i for i,e in enumerate(es) if h['doc_name']==e['file'] and m.squash(e['quote']) in m.squash(h['text'])} for h in pool]
        assert cov==seed, qid
        item={'qid':qid,'papers':n,'dense_retrieval':rank_metrics(cov,[set(range(n))]),
              'gold_matching_disagrees_with_seed_ids':cov!=seed,
              'paragraph_fragment_rule_differs_from_fact_quote':paragraph_cov!=cov,
              'pool_all_docs':int(set(q['gold_docs'])<={h['doc_name'] for h in pool}),
              'pool_doc_recall':len(set(q['gold_docs'])&{h['doc_name'] for h in pool})/n}
        for arm in ('one_pass','paper_read'):
            answer=read(cross/'items'/qid/f'{arm}.answer.json')
            assert answer['answer']==r[arm]['answer']
            if arm=='one_pass':
                assert len(answer['sources'])==len(pool)
                for source,hit in zip(answer['sources'],pool):
                    assert source['text']==hit['text'] and source['paper']==hit['doc_name']
            supported={i for i,e in enumerate(es) if any(s['paper']==e['file'] and m.squash(e['quote']) in m.squash(s['text']) for s in answer['sources'])}
            cv=r[arm]['coverage'];assert cv['status']=='evaluated'
            assert [p['requirement'] for p in cv['points']]==[p['requirement'] for p in r['oracle']['coverage']['points']]
            item[arm]={'context_evidence_hit':int(bool(supported)),
                       'context_evidence_recall':len(supported)/n,
                       'context_all_evidence':int(len(supported)==n),
                       'context_all_docs':int(set(q['gold_docs'])<={s['paper'] for s in answer['sources']}),
                       'correct':int(r[arm]['judge']['verdict']=='correct'),
                       'grounded':int(r[arm]['grounded']),
                       'answer_evidence_coverage':cv['ratio'],
                       'all_answer_points_supported':int(cv['ratio']==1)}
        cross_details.append(item)
    cross_summary={}
    for label,subset in [('two',[r for r in cross_details if r['papers']==2]),
                         ('three',[r for r in cross_details if r['papers']==3]),('all',cross_details)]:
        n=len(subset)
        result={'n':n,'dense_retrieval':agg([r['dense_retrieval'] for r in subset]),
                'all_docs_in_pool':sum(r['pool_all_docs'] for r in subset),
                'mean_pool_doc_recall':mean(r['pool_doc_recall'] for r in subset),
                'gold_matching_disagrees_with_seed_ids':sum(r['gold_matching_disagrees_with_seed_ids'] for r in subset)}
        for arm in ('one_pass','paper_read'):
            result[arm]=agg([r[arm] for r in subset])
            assert sum(r[arm]['correct'] for r in subset)==saved_summary[label][arm]['correct']
            assert round(result[arm]['answer_evidence_coverage'],4)==saved_summary[label][arm]['mean_evidence_coverage']
        cross_summary[label]=result
    history=read(ROOT/'evaluation/results/20261102-171638_qasper_ab_20261102_171451_A_dense.json')
    summary={'historical1309':history['overall'],'single500':single_summary,'cross93':cross_summary,
        'definitions':{'single':'original Qasper covers, max recall among annotator groups',
                       'cross':'one required fact quote per paper, file-scoped whitespace-normalized exact containment; verified identical to supporting_chunk_ids on all93 pools',
                       'answer_evidence_coverage':'mean per-question supported required points / required points; cached LLM audit',
                       'mrr5':'reciprocal rank of first marked evidence within5; does not measure multi-paper completeness'},
        'missing':['500 answer_evidence_coverage','93 dense-top5 answer correctness/coverage baseline',
                   '500 final paper_read evaluation','500 rerank top10/50 ranked output'],
        'calls':0}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (OUT/'single500-details.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in single_details))
    (OUT/'cross93-details.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in cross_details))
    hashes[str(ROOT/'evaluation/metrics.py')]=hashlib.sha256((ROOT/'evaluation/metrics.py').read_bytes()).hexdigest()
    hashes[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'provenance.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
