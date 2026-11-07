"""Replay the coverage audit on saved answers; no new retrieval or answer generation."""
import concurrent.futures
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = Path('/Users/bia/PycharmProjects/PythonProject5')
sys.path.append(str(ROOT))
from answer import Source
import evidence_coverage as module
from llm import DeepSeekClient

data = OUT.parent/'qasper-crosspaper-v1/paper-read/results-80000/pairs.jsonl'
rows = list(map(json.loads,data.read_text().splitlines()))
assert len(rows)==len({r['qid'] for r in rows})==16
items = OUT/'items'
items.mkdir(exist_ok=True)

def audit(r):
    c=DeepSeekClient(timeout=180)
    sources=[Source(s['n'],s['paper'],s['heading'],'',0,s['text']) for s in r['sources']]
    inputs=[r['question'],r['answer'],r['sources'],c.model,Path(module.__file__).read_text()]
    key=hashlib.sha256(json.dumps(inputs,ensure_ascii=False).encode()).hexdigest()
    path=items/(r['qid']+'.json')
    item=json.loads(path.read_text()) if path.exists() else {}
    if item.get('key')!=key or item['coverage']['status']=='unavailable':
        coverage=module.assess_coverage(r['question'],r['answer'],sources,c)
        item=dict(qid=r['qid'],key=key,question=r['question'],answer=r['answer'],coverage=asdict(coverage))
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(item,ensure_ascii=False,indent=2));tmp.replace(path)
    return item

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(audit,rows))
    lines=['# 跨论文问答 · 逐项证据覆盖度','',
           '直接回放既有回答和当时可见的原文。分母由问题单独生成，分子只统计回答中有有效引用、可定位原文且语义通过审核的要点。','',
           '|题号|已覆盖 / 必答|原实验答题判定|','|---|---:|---|']
    for r,v in zip(rows,results):
        c=v['coverage'];count=f"{c['covered']}/{c['total']}" if c['status']=='evaluated' else '审核不可用'
        print(r['qid'],count,r['grade']['verdict'])
        lines.append(f"| {r['qid']} | {count} | {r['grade']['verdict']} |")
    for v in results:
        c=v['coverage']; lines+=['','## '+v['qid'],'',v['question'],'',v['answer'],'']
        for item in c['points']:
            lines+=['- '+item['status']+' · '+item['requirement']+' — '+item['reason']]
            for e in item['evidence']:lines+=['  - ['+str(e['source'])+'] '+e['quote']]
        if c['note']: lines+=[c['note']]
    (OUT/'逐题证据覆盖.md').write_text('\n'.join(lines)+'\n')
