from pathlib import Path
from collections import Counter
import json

out = Path(__file__).resolve().parent
previous = out.parent / 'sc68-refusal-v2'
def rows(folder):
    data = [json.loads(s) for s in (folder/'pairs.jsonl').read_text().splitlines()]
    assert len(data) == len({r['qid'] for r in data}) == 68
    return {r['qid']: r for r in data}
old, new = rows(previous), rows(out)
assert old.keys() == new.keys()
special = ['1805.09821-q01','1809.08935-q02','1908.05925-q03']
def verdict(row, arm):
    return row[arm]['judge']['verdict']
for i,r in new.items():
    before=old[i]
    assert all(before[key]==r[key] for key in ('question','references','gold_evidence'))
    for arm in ('dense','rerank'):
        assert before[arm]['sources'] == r[arm]['sources']
        a=json.loads((out/'items'/i/f'{arm}.json').read_text())
        b=json.loads((previous/'items'/i/f'{arm}.json').read_text())
        assert a['model']==b['model'] and a['sha']!=b['sha']
regress=[k for k in old if verdict(old[k],'rerank')=='correct' and verdict(new[k],'rerank')!='correct']
assert sum(verdict(r,'rerank')=='correct' for r in old.values())==65
lines=['# SC68：固定前五段资料，比较两版回答提示词','',
       '按相同 qid、问题、参考答案及两条检索路径前五段的编号、正文和顺序逐项核对。每轮分别对 Dense 与重排的匿名回答评分，下表按 qid 比较两轮的模型判定。','',
       '| 检索方案 | 旧版正确 | 新版正确 | 对→对 | 对→错 | 错→对 | 错→错 | 涉及待定 |','|---|---:|---:|---:|---:|---:|---:|---:|']
for arm in ('dense','rerank'):
    c=Counter((verdict(old[k],arm),verdict(new[k],arm)) for k in old)
    cols=[sum(n for (a,b),n in c.items() if a=='correct'),sum(n for (a,b),n in c.items() if b=='correct'),*(c[a,b] for a,b in [('correct','correct'),('correct','incorrect'),('incorrect','correct'),('incorrect','incorrect')]),sum(n for key,n in c.items() if 'uncertain' in key)]
    lines.append('| '+' | '.join([arm,*map(str,cols)])+' |')
lines+=['','## 目标题、证据不足题与本轮回退','']
for i in dict.fromkeys(special+regress):
    r=new[i]; g=r['rerank']['judge']
    lines.extend([f'### {i} · {verdict(old[i],"rerank")} → {g["verdict"]}','',r['question'],'','参考答案：'+' / '.join(r['references']),'','新版回答：'+r['rerank']['answer'],'','引用判定：'+g['citation']+'。评审说明：'+g['why'],'','旧版回答：'+old[i]['rerank']['answer'],''])
lines+=['## 全部 68 题逐题对照','','| qid | 旧版重排组 | 新版重排组 | 旧版引用 | 新版引用 |','|---|---|---|---|---|']
for i in old:
    a,b=old[i]['rerank']['judge'],new[i]['rerank']['judge']
    lines.append(f"| {i} | {a['verdict']} | {b['verdict']} | {a['citation']} | {b['citation']} |")
lines+=['','[本轮逐题回答、来源原文和评审意见](review.html) · [上一轮报告](../sc68-refusal-v2/review.html)']
(out/'comparison.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[4:8])); print('regressions',regress)
