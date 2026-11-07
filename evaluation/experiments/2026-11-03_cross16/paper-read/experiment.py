"""Evaluate automatic paper selection with the saved, unfiltered cross-paper Dense pools."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'one-pass'))
import compare
from answer import generate_answer, cited_indexes
from llm import DeepSeekClient
from paper_reader import DEFAULT_BYTES, make_sources, paper_path, select_papers
import paper_reader
from vector_store import SearchHit

CORPUS=compare.PROJECT/'evaluation/data/qasper/corpus'
BENCH=HERE.parent
cached=compare.previous.cached
read=compare.previous.read


def run_question(q, folder, baseline, budget):
    ident=q['qid']
    dest=folder/'items'/ident
    dest.mkdir(parents=True,exist_ok=True)
    pool=[SearchHit(**h) for h in json.loads((BENCH/'run/items'/ident/'pool.json').read_text())]
    client=DeepSeekClient(timeout=180)
    def choose():
        start=time.perf_counter()
        names=select_papers(q['question'],pool,client)
        return dict(names=names,seconds=time.perf_counter()-start,
                    input_tokens=client.prompt_tokens,output_tokens=client.completion_tokens)
    selection=cached(dest/'selection.json',choose)
    sources,modes=make_sources(selection['names'],pool,CORPUS,budget)
    def answer():
        start=time.perf_counter()
        input_before,output_before=client.prompt_tokens,client.completion_tokens
        text=generate_answer(q['question'],sources,client)
        if not text.strip():
            raise ValueError(f'{ident} 生成空回答')
        return dict(answer=text,seconds=time.perf_counter()-start,
                    input_tokens=client.prompt_tokens-input_before,output_tokens=client.completion_tokens-output_before)
    generated=cached(dest/'answer.json',answer)
    material=[dict(n=s.index,paper=s.doc_name,heading=s.heading_path,text=s.text) for s in sources]
    arm=dict(answer=generated['answer'],sources=material)
    # Reference papers and reference claims are introduced only after generation.
    grades=cached(dest/'grades.json',lambda:compare.grade(q,dict(paper_read=arm,one_pass=baseline['one_pass'])))
    gold=set(q['gold_docs'])
    selected=set(selection['names'])
    cited=cited_indexes(generated['answer'])
    cited_files={s.doc_name for s in sources if s.index in cited}
    grounded=(grades['paper_read']['verdict']=='correct' and grades['paper_read']['citation']=='supported'
              and cited<={s.index for s in sources} and gold<=cited_files)
    return dict(qid=ident,question=q['question'],gold_docs=sorted(gold),pool_has_both=gold<={h.doc_name for h in pool},
                selected=selection['names'],selected_both=gold<=selected,modes=modes,answer=generated['answer'],
                sources=material,cited_files=sorted(cited_files),grade=grades['paper_read'],grounded=grounded,
                one_pass_rejudged=grades['one_pass'],input_tokens=selection['input_tokens']+generated['input_tokens'],
                output_tokens=selection['output_tokens']+generated['output_tokens'],
                seconds=selection['seconds']+generated['seconds'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--budget-bytes',type=int,default=DEFAULT_BYTES)
    args=p.parse_args()
    if args.budget_bytes<2048: p.error('两篇论文的总预算至少为 2048 字节')
    folder=HERE/f'results-{args.budget_bytes}'
    folder.mkdir(exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        qs=read(BENCH/'questions.jsonl')
        baselines={v['qid']:v for v in read(BENCH/'one-pass/pairs.jsonl')}
        assert len(qs)==len(baselines)==16
        files={}; corpus_names=set()
        for q in qs:
            pool_file=BENCH/'run/items'/q['qid']/'pool.json'
            pool=json.loads(pool_file.read_text())
            assert q['question']==baselines[q['qid']]['question'] and len({h['chunk_id'] for h in pool})==len(pool)==50
            files[str(pool_file)]=compare.digest(pool_file)
            corpus_names.update(h['doc_name'] for h in pool)
        for path in [Path(__file__), Path(paper_reader.__file__),Path(compare.__file__),BENCH/'run/evaluate.py',
                     BENCH/'one-pass/pairs.jsonl',BENCH/'questions.jsonl',compare.PROJECT/'answer.py',compare.PROJECT/'llm.py']:
            files[str(path)]=compare.digest(path)
        config=dict(files=files,corpus={n:compare.digest(paper_path(CORPUS,n)) for n in sorted(corpus_names)},
                    budget_bytes=args.budget_bytes,model=DeepSeekClient().model)
        if cached(folder/'config.json',lambda:config)!=config:
            raise ValueError('缓存与输入版本不一致，请将旧结果目录重命名后再运行')
        rows=[]
        with ThreadPoolExecutor(max_workers=3) as executor:
            for row in executor.map(lambda q:run_question(q,folder,baselines[q['qid']],args.budget_bytes),qs):
                rows.append(row);print(f"{len(rows)}/16 {row['qid']} {row['selected']} {row['grade']['verdict']}",flush=True)
        selected=[r for r in rows if r['selected_both']]
        passed=[r for r in rows if r['grade']['verdict']=='correct']
        old={i for i,v in baselines.items() if v['one_pass']['judge']['verdict']=='correct'}
        new={r['qid'] for r in passed}
        summary=dict(n=len(rows),pool_has_both=sum(r['pool_has_both'] for r in rows),selected_both=len(selected),
                     correct=len(passed),correct_given_both=sum(r['selected_both'] for r in passed),
                     grounded=sum(r['grounded'] for r in rows),questions_with_section_excerpts=sum('sections' in r['modes'].values() for r in rows),
                     historical_one_pass_correct=len(old),improved_vs_historical=sorted(new-old),lost_vs_historical=sorted(old-new),
                     one_pass_correct_in_current_grading=sum(r['one_pass_rejudged']['verdict']=='correct' for r in rows),
                     generation_input_tokens=sum(r['input_tokens'] for r in rows),generation_output_tokens=sum(r['output_tokens'] for r in rows))
        (folder/'pairs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        print(folder/'summary.json')


if __name__=='__main__':
    main()
