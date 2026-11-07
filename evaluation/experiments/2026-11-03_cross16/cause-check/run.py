from pathlib import Path
import sys,json,random,hashlib,concurrent.futures as cf
import pyarrow.parquet as pq
out=Path(__file__).resolve().parent
sys.path.insert(0,str(out.parent/'fulltext'))
import evaluate_fulltext as util
from answer import Source,build_prompt,generate_answer,cited_indexes,NOT_FOUND
from llm import DeepSeekClient
root=util.ROOT
assert hashlib.sha256((root/'answer.py').read_bytes()).hexdigest()==json.loads((out.parent/'run/run_config.json').read_text())['files']['answer.py']
qs={r['qid']:r for r in util.read(out.parent/'questions.jsonl')}
rag={r['qid']:r for r in util.read(out.parent/'run/pairs.jsonl')}
full=util.read(out.parent/'fulltext/pairs.jsonl')
selected=[r['qid'] for r in full if r['judge']['verdict']=='correct' and rag[r['qid']]['rerank']['judge']['verdict']=='incorrect']
assert len(selected)==9
idx=next((root/'evaluation/data/qasper/index/milvus.db/collections/notes/partitions/_default/data').glob('*.parquet'))
chunks={r['id']:r for r in pq.read_table(idx,columns=['id','doc_name','source_path','heading_path','text']).to_pylist()}
extra='逐篇核对所给段落，分别写出每项研究有证据的事实及其引用，再用一句话说明两者的关系。只对无法确认的部分说明缺口；不要依靠背景知识猜测。'
def process(i):
    q=qs[i];v=rag[i]['rerank']; folder=out.parent/'run/items'/i
    pool=json.loads((folder/'pool.json').read_text()); ids=[s['id'] for s in v['sources']]
    oracle=ids.copy();protect=set()
    for side in q['source_evidence']:
        matches=set(side['supporting_chunk_ids']);protect.update(matches)
    for side in q['source_evidence']:
        matches=side['supporting_chunk_ids']
        if not set(matches).intersection(oracle):
            pick=next((h['chunk_id'] for h in pool if h['chunk_id'] in matches),matches[0])
            pos=max(n for n,x in enumerate(oracle) if x not in protect)
            oracle[pos]=pick
    assert len(set(oracle))==5
    c=DeepSeekClient(timeout=120)
    conditions={}
    for arm,seq in [('replay',ids),('prompt_only',ids),('evidence_only',oracle)]:
        sources=[Source(k,s['doc_name'],s['heading_path'],s['source_path'],0,s['text']) for k,s in enumerate((chunks[x] for x in seq),1)]
        if arm=='prompt_only':
            class Override:
                def chat(self,p,**kw):return c.chat(extra+'\n'+p,**kw)
            client=Override()
        else:client=c
        a=generate_answer(q['question'],sources,client)
        conditions[arm]=dict(ids=seq,answer=a,cited_docs=sorted({s.doc_name for s in sources if s.index in cited_indexes(a)}))
    # Grade four anonymous answers in shuffled order, including the originally failed answer.
    conditions['historical']=dict(ids=ids,answer=v['answer'],cited_docs=sorted({s['paper'] for s in v['sources'] if s['n'] in cited_indexes(v['answer'])}))
    arms=list(conditions);random.Random(i).shuffle(arms)
    labels=dict(zip('WXYZ',arms))
    payload=dict(question=q['question'],reference=q['answers'][0],facts=[{k:r[k] for k in ['title','fact','text']} for r in q['source_evidence']],answers={l:{'answer':conditions[arm]['answer'],'inputs':[{'n':n,'title':chunks[x]['heading_path'],'text':chunks[x]['text']} for n,x in enumerate(conditions[arm]['ids'],1)]} for l,arm in labels.items()})
    instructions='独立评价每个匿名回答。必须同时给出两篇论文的必要事实及正确的比较，否则判错。拒答判错。支持性另判，两侧核心事实均需引用有效材料。接受等义答案。返回JSON，键为W,X,Y,Z，各含 correct 和 supported 布尔值及 why 中文原因。'
    for _ in range(3):
        j=c.chat_json(instructions+'\n'+json.dumps(payload,ensure_ascii=False),max_tokens=1700)
        if all(isinstance(j.get(k),dict) and type(j[k].get('correct'))==bool and type(j[k].get('supported'))==bool for k in labels):break
    else:raise ValueError(i)
    for k,arm in labels.items():conditions[arm]['judge']=j[k];conditions[arm]['grounded']=j[k]['correct'] and j[k]['supported'] and set(q['gold_docs'])<=set(conditions[arm]['cited_docs'])
    row=dict(qid=i,question=q['question'],conditions=conditions,swaps=sum(a!=b for a,b in zip(oracle,ids)))
    (out/(i+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2))
    print(i,[(arm,int(v['judge']['correct']),int(v['grounded'])) for arm,v in conditions.items()],flush=True)
with cf.ThreadPoolExecutor(max_workers=5) as ex:list(ex.map(process,selected))
