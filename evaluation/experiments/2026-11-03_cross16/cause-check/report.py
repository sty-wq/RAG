from pathlib import Path
import json,collections
out=Path(__file__).resolve().parent;root=out.parent
qs={r['qid']:r for r in map(json.loads,(root/'questions.jsonl').read_text().splitlines())}
rows=[json.loads(p.read_text()) for p in sorted(out.glob('cross-v1-*.json'))];assert len(rows)==9
arms=['historical','replay','prompt_only','evidence_only']
summary={a:{k:sum(r['conditions'][a]['judge']['correct'] if k=='correct' else r['conditions'][a]['grounded'] for r in rows) for k in ('correct','grounded')} for a in arms}
print(summary)
md=['# 9 道全文答对、RAG 答错题的对照','',
'选择同一轮 16 题中已出现差异的 9 题。使用原回答模型和生成函数，回放时从已建索引读取同样的段落。每组输入保持五段。补证据时，已命中的证据保持原位；每缺一侧，替换一条排在靠后的非标注证据块。若候选池内已有该证据，优先从池中取；否则从同一索引取。这个条件用标注诊断原因，线上检索没有这些标注。','',
'提示词组仅增加通用指令，要求逐篇核对事实、分别引用再综合且不猜测。旧回答及三种新回答打乱展示顺序共同评审，分别核对内容正确和引用充分。每题每条件只重新生成一次。','',
'| 条件 | 正确 | 正确且双侧引用充分 |','|---|---:|---:|']
for a in arms:md.append(f"| {a} | {summary[a]['correct']}/9 | {summary[a]['grounded']}/9 |")
for r in rows:
 i=r['qid'];q=qs[i];p=root/'run/items'/i;pool=json.loads((p/'pool.json').read_text()); top=json.loads((p/'top5.json').read_text())
 md.extend(['','## '+i,'',q['question'],'','| 来源 | 论文首见于 Dense 前50 | 标注证据首见于 Dense 前50 | 论文首见于重排前5 | 标注证据首见于重排前5 |','|---|---:|---:|---:|---:|'])
 for s in q['source_evidence']:
  ranks=[]
  for arr in [pool,top]:
   ranks.extend([next((str(n) for n,h in enumerate(arr,1) if h['doc_name']==s['file']),'—'),next((str(n) for n,h in enumerate(arr,1) if h['chunk_id'] in s['supporting_chunk_ids']),'—')])
  md.append('| '+s['title']+' | '+' | '.join(ranks)+' |')
 md+=['',f"替换段落数：{r['swaps']}。"]
 for a in arms:
  v=r['conditions'][a]
  md+=['','### '+a,'',v['answer'],'','评审：'+v['judge']['why']]
(out/'逐题归因.md').write_text('\n'.join(md)+'\n')
