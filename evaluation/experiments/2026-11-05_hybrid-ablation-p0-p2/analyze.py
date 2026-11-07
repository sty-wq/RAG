"""P0-P2 offline replay. All inputs frozen; no model or embedding calls.

Primary history reproduction is gated on complete Dense1309 cache. BM25 is
the exact frozen project implementation over the exact indexed text, sorted
by chunk_id as VectorStore.lexical_search does. Metrics use frozen helpers.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

OUT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(OUT/'frozen/code'))
sys.path.insert(0, str(OUT/'frozen/code/evaluation'))
from lexical import BM25Index
from metrics import covers, squash, hit_at_k_covered, recall_at_k_covered, reciprocal_rank_covered
import pyarrow.parquet as pq


def read(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def write(p, value):
    p.parent.mkdir(exist_ok=True, parents=True)
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    t.replace(p)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def chunks():
    manifest = json.loads((OUT/'manifest.json').read_text())
    for name, digest in manifest['frozen_files'].items():
        assert sha(OUT/name) == digest, name
    p = next((OUT/'frozen/milvus.db').rglob('*.parquet'))
    fields = ['id','doc_name','source_path','heading_path','chunk_order','text']
    rows = sorted(pq.read_table(p, columns=fields).to_pylist(), key=lambda r:r['id'])
    assert len(rows) == len({r['id'] for r in rows}) == 17887
    return rows


def bm25(qs, rows, dest):
    cfg = {'questions':hashlib.sha256(json.dumps(qs,ensure_ascii=False).encode()).hexdigest(),
           'lexical':sha(OUT/'frozen/code/lexical.py'), 'k':50, 'heading_plus_text':True,
           'chunks':hashlib.sha256(json.dumps(rows,ensure_ascii=False).encode()).hexdigest()}
    if dest.exists():
        assert json.loads(dest.with_suffix('.config.json').read_text()) == cfg
        result = read(dest)
        assert [r['qid'] for r in result] == [q['qid'] for q in qs]
        return {r['qid']:r['hits'] for r in result}
    index = BM25Index([r['heading_path']+'\n'+r['text'] for r in rows])
    result = []
    for i,q in enumerate(qs,1):
        result.append({'qid':q['qid'],'hits':[{'chunk_id':rows[j]['id'],'score':s}
                       for j,s in index.search(q['question'],50)]})
        if i%250 == 0: print('BM25',i,'/',len(qs),flush=True)
    dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result))
    write(dest.with_suffix('.config.json'),cfg)
    return {r['qid']:r['hits'] for r in result}


def weighted(dense, lexical, alpha):
    if alpha == 0:
        return list(dense)  # strict Dense ordering, including ties; no lexical-only tail.
    scores = {}
    for ranking, weight in ((dense,1.),(lexical,alpha)):
        assert len(ranking) == len(set(ranking))
        for rank, cid in enumerate(ranking,1):
            scores[cid] = scores.get(cid,0.) + weight/(60+rank)
    return sorted(scores,key=lambda cid:(-scores[cid],cid))


def judge(q, rankings, lookup):
    paras = sorted({p for g in q['evidence'] for p in g})
    groups = [{paras.index(p) for p in g} for g in q['evidence']]
    cov = {cid:{i for i,p in enumerate(paras) if covers(lookup[cid]['text'],p)}
           for cid in set().union(*map(set,rankings.values()))}
    scores = {}
    for name, ids in rankings.items():
        covered = [cov[c] for c in ids]
        m = {}
        for k in (1,5,10,50):
            m[f'hit@{k}'] = hit_at_k_covered(covered,k)
            m[f'recall@{k}'] = recall_at_k_covered(covered,groups,k)
        m['mrr@10'] = reciprocal_rank_covered(covered[:10])
        m['full_recall@50'] = float(m['recall@50'] == 1.)
        m['doc_hit@5'] = float(bool(set(q['gold_docs']) & {lookup[c]['doc_name'] for c in ids[:5]}))
        scores[name] = m
    return scores,cov,groups,paras


def aggregate(metrics):
    return {key:statistics.mean(m[key] for m in metrics) for key in metrics[0]}


def hist_audit(qs, lookup, lexical):
    histories = {tag:{r['qid']:r for r in read(OUT/f'frozen/history-{tag}.jsonl')}
                 for tag in ('A','B','C')}
    valid = [q for q in qs if q['should_answer'] and any(q['evidence'])]
    b_identical = b_score = 0
    recomputed = {tag:[] for tag in histories}
    regress = []
    improvements = []
    for q in valid:
        qid = q['qid']
        ranks = {tag:[h['chunk_id'] for h in data[qid]['hits']] for tag,data in histories.items()}
        scores,cov,_,_ = judge(q,ranks,lookup)
        for tag in histories:
            # Exact per-question historical metric reproduction from frozen indexed text.
            for key in ('hit@1','hit@5','recall@5','recall@10','doc_hit@5'):
                assert math.isclose(scores[tag][key],histories[tag][qid][key],abs_tol=1e-12),(qid,tag,key)
            assert math.isclose(scores[tag]['mrr@10'],histories[tag][qid]['mrr'],abs_tol=1e-12)
            recomputed[tag].append(scores[tag])
        now = lexical[qid][:10]
        b_identical += [h['chunk_id'] for h in now] == ranks['B']
        b_score += all(math.isclose(h['score'],old['score'],rel_tol=1e-12,abs_tol=1e-12)
                       for h,old in zip(now,histories['B'][qid]['hits'],strict=True))
        if scores['A']['hit@5'] and not scores['C']['hit@5']:
            top = histories['C'][qid]['hits'][:5]
            regress.append({'qid':qid,'question':q['question'],
                'doc_hit5':scores['C']['doc_hit@5'],
                'all_top5_common':all(h['dense_score'] is not None and h['lexical_score'] is not None for h in top),
                'dense_top5_relevant':[c for c in ranks['A'][:5] if cov[c]],
                'hybrid_top5':ranks['C'][:5]})
        if not scores['A']['hit@5'] and scores['C']['hit@5']:improvements.append(qid)
    result = {'n':len(valid),'history_metrics_from_current_index_exact':True,
        'bm25_top10_identical_questions':b_identical,'bm25_scores_match_questions':b_score,
        'metrics':{tag:{k:v for k,v in aggregate(ms).items() if '@50' not in k}
                   for tag,ms in recomputed.items()},
        'hit5_regressions':len(regress),'hit5_improvements':len(improvements),
        'regressions_all_top5_common':sum(r['all_top5_common'] for r in regress),
        'regressions_doc_hit5':sum(r['doc_hit5'] for r in regress),
        'improved_qids':improvements}
    write(OUT/'historical-audit.json',result)
    write(OUT/'historical-regressions.json',regress)
    print('Historical audit',json.dumps(result,ensure_ascii=False)[:1000],flush=True)


def evaluate(name, qs, lookup, lexical):
    dest = OUT/name
    dest.mkdir(exist_ok=True)
    scored = [q for q in qs if q.get('should_answer',True) and any(q['evidence'])]
    old = {tag:{r['qid']:r for r in read(OUT/f'frozen/history-{tag}.jsonl')}
           for tag in ('A','B','C')} if name == 'original1309' else None
    records=[]
    p0=Counter()
    p0_details=[]
    p0_metrics={tag:[] for tag in ('A','B','C')}
    for n,q in enumerate(scored,1):
        qid=q['qid']
        path = OUT/('dense1309' if old else 'dense500')/f'{qid}.json'
        data=json.loads(path.read_text())
        dh = data['hits50'] if old else data
        if old: assert data['question'] == q['question']
        dense=[h['chunk_id'] for h in dh]
        lex=[h['chunk_id'] for h in lexical[qid]]
        assert len(dense)==len(set(dense))==50
        assert len(lex)==len(set(lex))==50
        for h in dh:
            for key in ('text','heading_path','doc_name','chunk_order'):
                assert h[key] == lookup[h['chunk_id']][key],(qid,key)
        ranks={'dense':dense,'bm25':lex}
        for alpha in (0,.25,.5,1):
            ranks[f'rrf_a{alpha:g}']=weighted(dense,lex,alpha)[:50]
        assert ranks['rrf_a0']==dense
        union=list(dict.fromkeys(dense+lex))
        scores,cov,groups,paras=judge(q,{**ranks,'union':union},lookup)
        # Union coverage uses *all* up to100 candidates, never union[:50].
        union_cov=[cov[c] for c in union]
        upper={'count':len(union),'hit':hit_at_k_covered(union_cov,len(union)),
               'recall':recall_at_k_covered(union_cov,groups,len(union))}
        del scores['union']
        if old:
            current10=[h['chunk_id'] for h in data['hits10']]
            p0['dense_k10_vs_k50_prefix_identical'] += current10==dense[:10]
            measured,_,_,_=judge(q,{'A':current10,'B':lex[:10],'C':ranks['rrf_a1'][:10]},lookup)
            for tag,ids in [('A',current10),('B',lex[:10]),('C',ranks['rrf_a1'][:10])]:
                oldids=[h['chunk_id'] for h in old[tag][qid]['hits']]
                p0[f'{tag}_top10_identical'] += ids==oldids
                p0_metrics[tag].append(measured[tag])
                if ids != oldids:
                    p0_details.append({'qid':qid,'arm':tag,'old_ids':oldids,'new_ids':ids,
                        'old_scores':[h['score'] for h in old[tag][qid]['hits']],
                        'new_dense_scores':{h['chunk_id']:h['score'] for h in data['hits10']} if tag=='A' else None,
                        'same_set':set(ids)==set(oldids),
                        'metric_differences':{k:measured[tag][k]-old[tag][qid]['mrr' if k=='mrr@10' else k]
                            for k in ('hit@1','hit@5','recall@5','recall@10','mrr@10','doc_hit@5')
                            if measured[tag][k]!=old[tag][qid]['mrr' if k=='mrr@10' else k]}})
            p0['dense50_prefix_vs_historical_A'] += dense[:10]==[h['chunk_id'] for h in old['A'][qid]['hits']]
        dset,bset=set(dense),set(lex)
        common=dset&bset
        categories={'dense_only':dset-bset,'bm25_only':bset-dset,'common':common}
        relevant={tag:sorted(c for c in ids if cov[c]) for tag,ids in categories.items()}
        dc=set().union(*(cov[c] for c in dense));bc=set().union(*(cov[c] for c in lex))
        fused=ranks['rrf_a1'];fs=set(fused)
        common_before_exclusive=all(c in common for c in fused[:min(len(common),50)])
        assert common_before_exclusive
        first_gold=next((c for c in dense if cov[c]),None)
        full_rrf=weighted(dense,lex,1)
        drift={'first_dense_gold':first_gold,
            'first_dense_gold_dense_rank':dense.index(first_gold)+1 if first_gold else None,
            'first_dense_gold_bm25_rank':lex.index(first_gold)+1 if first_gold in bset else None,
            'first_dense_gold_fused_rank_before_truncation':full_rrf.index(first_gold)+1 if first_gold else None,
            'dense_gold_chunks_dropped_from50':sorted(c for c in dset-fs if cov[c]),
            'union_gold_chunks_dropped_from50':sorted(c for c in set(union)-fs if cov[c]),
            'dense_unique_evidence_units':sorted(dc-bc),'bm25_unique_evidence_units':sorted(bc-dc),
            'common_evidence_units':sorted(dc&bc)}
        drift['wrong_top5_intruders']=[{'chunk_id':c,
            'dense_rank':dense.index(c)+1 if c in dset else None,
            'bm25_rank':lex.index(c)+1 if c in bset else None,
            'heading_path':lookup[c]['heading_path']}
            for c in fused[:5] if c not in dense[:5] and not cov[c]]
        records.append({'qid':qid,'question':q['question'],'group':q.get('group',q['type']),
            'gold_docs':q['gold_docs'],'evidence':paras,'scores':scores,'union':upper,
            'rankings':ranks,'dense_scores':{h['chunk_id']:h['score'] for h in dh},
            'bm25_scores':{h['chunk_id']:h['score'] for h in lexical[qid]},
            'covered':{c:sorted(v) for c,v in cov.items() if v},
            'relevant_categories':relevant,'common_candidates':len(common),
            'top5_all_common':all(c in common for c in fused[:5]),'drift':drift})
        if n%250==0:print(name,'scored',n,'/',len(scored),flush=True)
    assert records
    names=list(records[0]['scores'])
    metrics={arm:aggregate([r['scores'][arm] for r in records]) for arm in names}
    reg=[r for r in records if r['scores']['dense']['hit@5'] and not r['scores']['rrf_a1']['hit@5']]
    gain=[r for r in records if not r['scores']['dense']['hit@5'] and r['scores']['rrf_a1']['hit@5']]
    effect={}
    for arm in names:
        effect[arm]={
            'vs_dense_hit5_improved':sum(r['scores'][arm]['hit@5']>r['scores']['dense']['hit@5'] for r in records),
            'vs_dense_hit5_regressed':sum(r['scores'][arm]['hit@5']<r['scores']['dense']['hit@5'] for r in records),
            'restored_current_hybrid_losses':sum(r['scores'][arm]['hit@5'] for r in reg),
            'retained_current_hybrid_gains':sum(r['scores'][arm]['hit@5'] for r in gain),
        }
    p1={
        'union_mean_size':statistics.mean(r['union']['count'] for r in records),
        'union_hit':statistics.mean(r['union']['hit'] for r in records),
        'union_recall':statistics.mean(r['union']['recall'] for r in records),
        'union_full_recall_questions':sum(r['union']['recall']==1 for r in records),
        'union_recall_gt_dense50_questions':sum(r['union']['recall']>r['scores']['dense']['recall@50'] for r in records),
        'fusion_recall50_lt_union_questions':sum(r['scores']['rrf_a1']['recall@50']<r['union']['recall'] for r in records),
        'fusion_recall50_lt_dense50_questions':sum(r['scores']['rrf_a1']['recall@50']<r['scores']['dense']['recall@50'] for r in records),
        'fusion_hit50_lt_dense50_questions':sum(r['scores']['rrf_a1']['hit@50']<r['scores']['dense']['hit@50'] for r in records),
        'categories':{tag:{'questions_with_relevant_chunk':sum(bool(r['relevant_categories'][tag]) for r in records),
                            'relevant_chunks_across_questions':sum(len(r['relevant_categories'][tag]) for r in records)}
                        for tag in ('dense_only','bm25_only','common')},
        'questions_with_bm25_unique_evidence':sum(bool(r['drift']['bm25_unique_evidence_units']) for r in records),
        'questions_with_dense_unique_evidence':sum(bool(r['drift']['dense_unique_evidence_units']) for r in records),
        'top5_all_common_questions':sum(r['top5_all_common'] for r in records),
        'mean_common_candidates':statistics.mean(r['common_candidates'] for r in records),
        'hit5_regressions':len(reg),'hit5_gains':len(gain),
        'regressions_still_hit50':sum(r['scores']['rrf_a1']['hit@50'] for r in reg),
        'regressions_no_hit50':sum(not r['scores']['rrf_a1']['hit@50'] for r in reg),
        'regressions_first_dense_gold_is_dense_only':sum(r['drift']['first_dense_gold_bm25_rank'] is None for r in reg),
        'regressions_first_dense_gold_also_in_bm25':sum(r['drift']['first_dense_gold_bm25_rank'] is not None for r in reg),
        'regressions_first_dense_gold_beyond50':sum(r['drift']['first_dense_gold_fused_rank_before_truncation']>50 for r in reg),
        'regressions_doc_hit5':sum(r['scores']['rrf_a1']['doc_hit@5'] for r in reg),
        'regressions_first_gold_median_dense_rank':statistics.median(r['drift']['first_dense_gold_dense_rank'] for r in reg) if reg else None,
        'regressions_first_gold_median_fused_rank':statistics.median(r['drift']['first_dense_gold_fused_rank_before_truncation'] for r in reg) if reg else None,
        'regressions_common_first_gold_median_bm25_rank':statistics.median(r['drift']['first_dense_gold_bm25_rank'] for r in reg if r['drift']['first_dense_gold_bm25_rank'] is not None) if any(r['drift']['first_dense_gold_bm25_rank'] is not None for r in reg) else None,
    }
    summary={'n':len(records),'dataset':name,'p0':dict(p0),'metrics':metrics,'p1':p1,'p2':effect,
        'by_group':{group:{'n':sum(r['group']==group for r in records),
                           'metrics':{arm:aggregate([r['scores'][arm] for r in records if r['group']==group]) for arm in names}}
                    for group in sorted({r['group'] for r in records})},
        'provenance':{'manifest_sha256':sha(OUT/'manifest.json'),'analysis_sha256':sha(Path(__file__)),
                      'dense_cache_hashes':{r['qid']:sha(OUT/('dense1309' if old else 'dense500')/f"{r['qid']}.json") for r in records}},
        'cautions':['Metrics follow original evidence substring/fragment matching, not answer correctness.',
                    'Alpha comparison is diagnostic on this set, not a held-out optimization result.',
                    'Upper-bound union has up to100 candidates; final systems use50.',
                    'Current input hashes cannot retroactively prove historical corpus/query immutability.']}
    if old:
        keys=('hit@1','hit@5','recall@5','recall@10','mrr@10','doc_hit@5')
        summary['p0_history_comparison']={tag:{
            'old':{k:statistics.mean(old[tag][q['qid']]['mrr' if k=='mrr@10' else k] for q in scored) for k in keys},
            'new':{k:statistics.mean(m[k] for m in p0_metrics[tag]) for k in keys}}
            for tag in ('A','B','C')}
        historic_reg=[r for r in records if old['A'][r['qid']]['hit@5'] and not old['C'][r['qid']]['hit@5']]
        historic_gain=[r for r in records if not old['A'][r['qid']]['hit@5'] and old['C'][r['qid']]['hit@5']]
        summary['historical_transitions_rechecked']={
            'historical_regressions':len(historic_reg),'historical_gains':len(historic_gain),
            'regression_qid_set_identical':{r['qid'] for r in historic_reg}=={r['qid'] for r in reg},
            'gain_qid_set_identical':{r['qid'] for r in historic_gain}=={r['qid'] for r in gain},
            'historical_losses_with_fresh_hybrid_hit50':sum(r['scores']['rrf_a1']['hit@50'] for r in historic_reg),
            'alpha_restored_historical_losses':{arm:sum(r['scores'][arm]['hit@5'] for r in historic_reg) for arm in names},
            'alpha_retained_historical_gains':{arm:sum(r['scores'][arm]['hit@5'] for r in historic_gain) for arm in names}}
        write(dest/'reproduction-differences.json',p0_details)
    write(dest/'summary.json',summary)
    (dest/'details.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    write(dest/'regressed-qids.json',[r['qid'] for r in reg])
    write(dest/'improved-qids.json',[r['qid'] for r in gain])
    write_report(dest,summary,records,lookup)
    print(name,json.dumps({'n':len(records),'p0':dict(p0),'p1':p1,'p2':effect},ensure_ascii=False),flush=True)


def write_report(dest,s,records,lookup):
    p=s['p1'];n=s['n']
    lines=[f"# 混合检索消融 P0–P2 · {s['dataset']}", '',
        f'计分题数 {n}。冻结同一题目、17887 块、两路各 50 候选。RRF 常数 60，无重排、无回答模型。', '',
        '## P0 复现', '', '```json',json.dumps(s['p0'],ensure_ascii=False,indent=2),'```','',
        '原始 1309 题要核对 A/B/C 的逐题前十顺序。500 题为既有 Dense50 缓存的补充回放，不能替代原始实验复现。','',
        '## 指标', '', '| 方法 | Hit@1 | Hit@5 | Recall@5 | Recall@10 | MRR@10 | Hit@50 | Recall@50 | Recall@50=1 题数 |',
        '|---|---|---|---|---|---|---|---|---|']
    for arm,m in s['metrics'].items():
        vals=[m[k] for k in ('hit@1','hit@5','recall@5','recall@10','mrr@10','hit@50','recall@50')]
        lines.append('| '+arm+' | '+' | '.join(f'{v:.4f}' for v in vals)+f" | {round(n*m['full_recall@50'])} |")
    if 'p0_history_comparison' in s:
        lines += ['', '历史与本次复现的聚合指标及逐题顺序差异独立记录，不能将近似同分换序说成完全相同。', '',
                  '```json',json.dumps(s['p0_history_comparison'],ensure_ascii=False,indent=2),'```', '',
                  f'[历史顺序差异明细]({dest}/reproduction-differences.json)', '']
    lines += ['',f"并集上限（平均 {p['union_mean_size']:.2f} 条，最多 100 条）：Hit={p['union_hit']:.4f}，Recall={p['union_recall']:.4f}，全覆盖 {p['union_full_recall_questions']}/{n}。这不是同预算系统。",'',
        '## P1 丢失位置', '',
        f"Dense→当前等权 RRF，Hit@5 变差 {p['hit5_regressions']} 题，变好 {p['hit5_gains']} 题。变差题中 {int(p['regressions_still_hit50'])} 题前 50 仍有证据，{p['regressions_no_hit50']} 题前 50 也无证据。",'',
        f"变差题中，Dense 最靠前正确块：{p['regressions_first_dense_gold_is_dense_only']} 题未进 BM25 前50；{p['regressions_first_dense_gold_also_in_bm25']} 题同时进入两路；{p['regressions_first_dense_gold_beyond50']} 题该块被挤出融合前50。",'',
        f"全部题中，并集相对 Dense50 提高 Recall 的有 {p['union_recall_gt_dense50_questions']} 题；融合截断相对并集损失 Recall 的有 {p['fusion_recall50_lt_union_questions']} 题；融合50相对 Dense50 的 Recall 下降 {p['fusion_recall50_lt_dense50_questions']} 题。",'',
        f"有 BM25 独有证据单元的题 {p['questions_with_bm25_unique_evidence']}；有 Dense 独有证据单元的题 {p['questions_with_dense_unique_evidence']}。独有块不一定代表独有证据，多个块可能覆盖同一段。",'',
        '| 相关块类别 | 有相关块的题数 | 各题相关块总数 |','|---|---|---|']
    for tag,x in p['categories'].items():lines.append(f"| {tag} | {x['questions_with_relevant_chunk']} | {x['relevant_chunks_across_questions']} |")
    lines += ['', '## P2 固定候选只改权重', '',
        '`score = 1/(60+dense_rank) + alpha/(60+bm25_rank)`，未命中一路时该路贡献为 0。alpha=0 严格沿用 Dense 顺序。其他权重用 chunk_id 打破同分，alpha=1 与生产 RRF 相同。','',
        '| 方法 | 相对Dense变好 | 相对Dense变差 | 恢复当前混合损失 | 保留当前混合收益 |', '|---|---|---|---|---|']
    for arm,x in s['p2'].items():lines.append(f"| {arm} | {x['vs_dense_hit5_improved']} | {x['vs_dense_hit5_regressed']} | {int(x['restored_current_hybrid_losses'])}/{p['hit5_regressions']} | {int(x['retained_current_hybrid_gains'])}/{p['hit5_gains']} |")
    lines += ['', '## 已证明的融合性质', '',
        '共同候选的最低分为 2/(60+50)=0.01818；单路独有候选的最高分为 1/(60+1)=0.01639。故所有共同候选都优先于所有独有候选。这是当前常数和候选深度共同造成的数学性质，并非所有 RRF 配置都如此。', '',
        f"本组 {p['top5_all_common_questions']}/{n} 题的融合前五全部为共同候选。它说明共同命中受到强奖励，不能单独证明每个错误都由这个机制引起。",'',
        '另外，当 alpha < 61/110≈0.5545 时，BM25 独有候选的最高分仍低于 Dense 第50名的最低分。因此本轮 alpha=.25/.5 的融合前50候选集合必然与 Dense50 相同，只改变内部顺序；其收益不能解释成补回了 BM25 独有证据。', '',
        '## 边界', '',
        '本轮只诊断候选互补性、融合排序和截断。标题、分词、长度归一化等 BM25 原因尚未做 P3 对照，不作因果定论。Recall 沿用历史匹配规则，长段落的足够长片段也可能计为覆盖。权重尚未通过独立留出集验证。','',
        f'[逐题候选、证据与排名]({dest}/details.jsonl) · [机器可读汇总]({dest}/summary.json) · [错误样本]({dest}/cases.md)']
    (dest/'报告.md').write_text('\n'.join(lines)+'\n')
    # Representative regression examples from both overlap categories, plus a lexical gain.
    regress=[r for r in records if r['scores']['dense']['hit@5'] and not r['scores']['rrf_a1']['hit@5']]
    chosen=[]
    for want_only in (True,False):
        options=[r for r in regress if (r['drift']['first_dense_gold_bm25_rank'] is None)==want_only]
        chosen += sorted(options,key=lambda r:(-r['drift']['first_dense_gold_fused_rank_before_truncation'],r['qid']))[:2]
    chosen += [r for r in records if not r['scores']['dense']['hit@5'] and r['scores']['rrf_a1']['hit@5']][:2]
    lines=['# 代表性错误与互补样本','', '样本按排名退化幅度选取，并补充 BM25 带来收益的题。判断“正确块”沿用金标准段落匹配，不代表人工回答判定。','']
    for r in chosen:
        lines += [f"## {r['qid']}",'',r['question'],'',
                  '| 方法 | Hit@5 | Recall@5 |','|---|---|---|']
        for arm,m in r['scores'].items():lines.append(f"| {arm} | {m['hit@5']:.0f} | {m['recall@5']:.4f} |")
        lines += ['', '| chunk_id | Dense名次 | BM25名次 | 等权RRF名次 | α=.25名次 | 覆盖证据编号 |', '|---|---|---|---|---|---|']
        ids=list(dict.fromkeys(r['rankings']['dense'][:5]+r['rankings']['rrf_a1'][:5]))
        full=weighted(r['rankings']['dense'],r['rankings']['bm25'],1)
        a25=weighted(r['rankings']['dense'],r['rankings']['bm25'],.25)
        for cid in ids:
            rr=[str(arr.index(cid)+1) if cid in arr else '未进池' for arr in (r['rankings']['dense'],r['rankings']['bm25'],full,a25)]
            lines.append('| '+cid+' | '+' | '.join(rr)+' | '+str(r['covered'].get(cid,[]))+' |')
        for cid in ids:
            h=lookup[cid]
            lines += ['',f"### {cid}",'',h['heading_path'],'',h['text'],'']
        lines += ['### 金标准证据','']
        for i,p in enumerate(r['evidence']):lines += [f'证据 {i}', '',p,'']
    (dest/'cases.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['all','original','rewritten'],default='all')
    parser.add_argument('--bm25-only',action='store_true')
    args=parser.parse_args()
    rows=chunks();lookup={r['id']:r for r in rows}
    if not (OUT/'chunks.jsonl').exists():
        (OUT/'chunks.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    if args.dataset in ('all','original'):
        qs=read(OUT/'frozen/questions_ctx.jsonl')
        lex=bm25(qs,rows,OUT/'bm25-original.jsonl')
        hist_audit(qs,lookup,lex)
        missing=[q['qid'] for q in qs if q['should_answer'] and any(q['evidence'])
                 and not (OUT/'dense1309'/f"{q['qid']}.json").exists()]
        if not args.bm25_only:
            if missing:print(f'原1309题等待 Dense 缓存，还缺 {len(missing)} 题。未生成原题融合消融结果。',flush=True)
            else:evaluate('original1309',qs,lookup,lex)
    if args.dataset in ('all','rewritten'):
        qs=read(OUT/'frozen/questions500.jsonl')
        lex=bm25(qs,rows,OUT/'bm25-rewritten.jsonl')
        if not args.bm25_only:evaluate('rewritten500',qs,lookup,lex)


if __name__=='__main__':
    main()
