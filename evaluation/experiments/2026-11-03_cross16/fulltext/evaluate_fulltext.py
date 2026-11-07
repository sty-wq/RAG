from pathlib import Path
import json, hashlib, random, sys
from concurrent.futures import ThreadPoolExecutor
OUT=Path(__file__).resolve().parent
ROOT=Path('/Users/bia/PycharmProjects/PythonProject5')
sys.path.insert(0,str(ROOT))
from llm import DeepSeekClient
from answer import Source,build_prompt,generate_answer,cited_indexes
def sha(x):return hashlib.sha256(x).hexdigest()
def read(p):return list(map(json.loads,p.read_text().splitlines()))
qs=read(OUT.parent/'questions.jsonl')
assert len(qs)==len({q['qid'] for q in qs})==16
def checkpoint(path,key,fn):
    if path.exists():
        item=json.loads(path.read_text())
        assert item['key']==key
    else:
        item=dict(key=key,**fn())
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(item,indent=2,ensure_ascii=False));tmp.replace(path)
    return item
RUBRIC="""独立审核匿名科研回答，仅依据所附材料与题意。分别判断：(a) 第一篇指定论文的必要事实是否正确、归属正确 (b) 第二篇同理 (relation) 是否明确或等价回答了两篇之间所问的关系。按给定验收要点，但应接受真正等价、符合题意的表述。不要因多了有原文支持的解释扣分。问题明确要求的量值、方向或单位必须正确；仅复述问题、遗漏其中一侧、实质错误、拒答不能判对。把引用支持与答案正确性分开：材料编号须和实际来源对应，两侧关键断言都要有支撑。若参考答案或问题本身有歧义明确指出，不强行判错。输出JSON对象，a,b,relation,citation_supported四个字段均为布尔值；verdict 取correct/incorrect/uncertain；why 用中文说明对应事实，缺什么要具体，绝不照抄参考答案当作被测回答。"""
def evaluate(q):
    cid=q['qid']; c=DeepSeekClient(timeout=180)
    docs=list(q['gold_docs']);random.Random(cid).shuffle(docs)
    sources=[]
    for i,name in enumerate(docs,1):
        p=ROOT/'evaluation/data/qasper/corpus'/name;raw=p.read_bytes()
        assert sha(raw)==next(r['file_sha256'] for r in q['source_evidence'] if r['file']==name)
        s=raw.decode('utf-8')
        sources.append(Source(i,name,s.splitlines()[0].removeprefix('# '),name,0,s))
    prompt=build_prompt(q['question'],sources)
    assert all(s.text in prompt for s in sources)
    def gen():
        ans=generate_answer(q['question'],sources,c)
        assert ans
        return dict(answer=ans,model=c.model,docs=docs,bytes=sum(len(s.text.encode()) for s in sources),input_tokens=c.prompt_tokens,calls=c.calls)
    answer=checkpoint(OUT/'items'/f'{cid}.answer.json',sha((c.model+prompt).encode()),gen)
    obj=dict(question=q['question'],candidate=answer['answer'],facts=[dict(paper=r['title'],source_number=docs.index(r['file'])+1,required=r['fact'],paragraph=r['text']) for r in q['source_evidence']],synthesis=q['synthesis'],reference=q['answers'],materials=[dict(n=s.index,title=s.heading_path,text=s.text) for s in sources])
    jp=RUBRIC+'\n'+json.dumps(obj,ensure_ascii=False)
    def judge():
        for _ in range(3):
            j=c.chat_json(jp,max_tokens=1000)
            if all(type(j.get(k)) is bool for k in ('a','b','relation','citation_supported')) and j.get('verdict') in ('correct','incorrect','uncertain') and (j['verdict']!='correct' or all(j[k] for k in ('a','b','relation'))) and isinstance(j.get('why'),str):return dict(grade=j)
        raise ValueError(cid+' invalid judge verdict')
    j=checkpoint(OUT/'items'/f'{cid}.judge.json',sha((c.model+jp).encode()),judge)['grade']
    return dict(qid=cid,question=q['question'],**answer,judge=j,both_cited=cited_indexes(answer['answer'])=={1,2})
if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=4) as ex:
        result=[]
        for r in ex.map(evaluate,qs):
            result.append(r);print(r['qid'],r['judge']['verdict'],flush=True)
    (OUT/'pairs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result))
