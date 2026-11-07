"""Compare two answering paths using identical saved Dense candidates."""
import concurrent.futures
import fcntl
import hashlib
import json
from pathlib import Path
import random
from statistics import median
import sys
import time

OUT = Path(__file__).resolve().parent
PROJECT = Path('/Users/bia/PycharmProjects/PythonProject5')
BASE = OUT.parent / 'run'
sys.path[:0] = [str(PROJECT), str(BASE)]
import evaluate as previous
from answer import answer_from_candidates, generate_answer, sources_from_hits, cited_indexes
from llm import DeepSeekClient
from reranker import DeepSeekReranker
from retrieval import rerank_hits
from vector_store import SearchHit


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(arm, question, pool):
    client = DeepSeekClient(timeout=150)
    start = time.perf_counter()
    if arm == 'one_pass':
        result = answer_from_candidates(question, pool, client)
        text, sources = result.answer, result.sources
    else:
        hits = rerank_hits(question, pool, DeepSeekReranker(client=client), 5)
        sources = sources_from_hits(hits)
        text = generate_answer(question, sources, client)
    return dict(answer=text, seconds=time.perf_counter()-start, calls=client.calls,
                input_tokens=client.prompt_tokens, output_tokens=client.completion_tokens,
                sources=[dict(n=s.index, paper=s.doc_name, heading=s.heading_path, text=s.text) for s in sources])


def grade(q, arms):
    labels = list(arms)
    random.Random(q['qid']).shuffle(labels)
    order = dict(zip('AB', labels))
    payload = dict(question=q['question'], references=q['answers'], synthesis=q['synthesis'],
                   evidence_by_paper=[{k:ref[k] for k in ('file','title','text','fact')} for ref in q['source_evidence']],
                   outputs={label:{k:arms[name][k] for k in ('answer','sources')} for label,name in order.items()})
    prompt = previous.rubric + '\n' + json.dumps(payload, ensure_ascii=False)
    client = DeepSeekClient(timeout=150)
    for limit in (1100, 1800, 2600):
        raw = client.chat(prompt, json_mode=True, max_tokens=limit)
        try:
            grades = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(grades,dict) or set(grades)!=set(order):
            continue
        def valid(g):
            return (isinstance(g,dict) and g.get('verdict') in ('correct','incorrect','uncertain')
                    and g.get('citation') in ('supported','incomplete','unsupported','na')
                    and all(type(g.get(k)) is bool for k in ('fact_a','fact_b','synthesis'))
                    and (g['verdict']!='correct' or all(g[k] for k in ('fact_a','fact_b','synthesis')))
                    and isinstance(g.get('why'),str) and g['why'].strip())
        if all(valid(g) for g in grades.values()):
            return {arm:grades[label] for label,arm in order.items()}
    raise ValueError(f"{q['qid']} 评审未返回有效结构")


def evaluate(q):
    folder = OUT/'items'/q['qid']
    folder.mkdir(parents=True, exist_ok=True)
    pool = [SearchHit(**h) for h in json.loads((BASE/'items'/q['qid']/'pool.json').read_text())]
    assert len(pool)==len({h.chunk_id for h in pool})==50
    arms = {}
    for name in ('two_step','one_pass'):
        arms[name] = previous.cached(folder/f'{name}.json', lambda:generate(name,q['question'],pool))
    judges = previous.cached(folder/'grades.json',lambda:grade(q,arms))
    for name,data in arms.items():
        data['judge']=judges[name]
        used=cited_indexes(data['answer'])
        valid_ids={s['n'] for s in data['sources']}
        docs={s['paper'] for s in data['sources'] if s['n'] in used}
        data['grounded']=(judges[name]['verdict']=='correct' and judges[name]['citation']=='supported'
                          and used<=valid_ids and set(q['gold_docs'])<=docs)
    return dict(qid=q['qid'],question=q['question'],**arms)


def main():
    with (OUT/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        qs=previous.qs
        history={r['qid']:r for r in previous.read(BASE/'pairs.jsonl')}
        assert len(history)==len(qs)==16 and all(history[q['qid']]['question']==q['question'] for q in qs)
        fingerprint=dict(model=DeepSeekClient().model, benchmark=digest(OUT.parent/'questions.jsonl'),
                         history=digest(BASE/'pairs.jsonl'), pools={q['qid']:digest(BASE/'items'/q['qid']/'pool.json') for q in qs},
                         code={str(p):digest(p) for p in [Path(__file__),BASE/'evaluate.py',PROJECT/'answer.py',PROJECT/'llm.py',PROJECT/'reranker.py',PROJECT/'retrieval.py']})
        if previous.cached(OUT/'config.json',lambda:fingerprint)!=fingerprint:
            raise ValueError('输入、模型或代码已改变；请为本轮使用新的同级输出目录')
        rows=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            for r in executor.map(evaluate,qs):
                rows.append(r)
                print(f"{len(rows)}/{len(qs)} {r['qid']}",flush=True)
        def correct(row):return row['judge']['verdict']=='correct'
        summary={}
        for arm in ('two_step','one_pass'):
            values=[r[arm] for r in rows]
            summary[arm]=dict(correct=sum(map(correct,values)), grounded=sum(v['grounded'] for v in values),
                              p50_seconds=round(median(v['seconds'] for v in values),3),
                              **{k:sum(v[k] for v in values) for k in ('input_tokens','output_tokens','calls')})
        old_good={i for i,v in history.items() if correct(v['rerank'])}
        targets={i for i,v in history.items() if not correct(v['rerank']) and v['pool50']['both_evidence']}
        good={r['qid'] for r in rows if correct(r['one_pass'])}
        summary.update(n=len(rows),recovered_of_historical_five=sorted(good & targets),
                       lost_from_historical_six=sorted(old_good-good),
                       improved_vs_fresh_two_step=[r['qid'] for r in rows if correct(r['one_pass']) and not correct(r['two_step'])],
                       regressed_vs_fresh_two_step=[r['qid'] for r in rows if correct(r['two_step']) and not correct(r['one_pass'])])
        (OUT/'pairs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        text=json.dumps(summary,indent=2,ensure_ascii=False)
        (OUT/'summary.json').write_text(text+'\n')
        print(text)

if __name__=='__main__':
    main()
