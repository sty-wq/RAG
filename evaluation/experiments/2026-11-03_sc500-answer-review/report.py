from collections import Counter
from pathlib import Path
from html import escape
import json
out=Path(__file__).resolve().parent
rows=[json.loads(s) for s in (out/'pairs.jsonl').read_text().splitlines()]
assert len(rows)==len({r['qid'] for r in rows})==500
def transition(r):
    a,b=[r[k]['judge']['verdict'] for k in ('dense','rerank')]
    if 'uncertain' in (a,b): return '待复核'
    return {('incorrect','correct'):'错→对',('correct','incorrect'):'对→错',('incorrect','incorrect'):'两组均错',('correct','correct'):'两组均对'}[a,b]
def e(x): return escape(str(x))
counts=Counter(map(transition,rows))
totals={k:Counter(r[k]['judge']['verdict'] for r in rows) for k in ['dense','rerank']}
print(dict(counts),totals)
def answer(r,k):
    v=r[k]; j=v['judge']
    items=''.join('<li><b>'+e(s['heading'])+'</b><br>'+e(s['text'])+'</li>' for s in v['sources'])
    return f"<section><h3>{'Dense' if k=='dense' else 'Dense + DeepSeek 重排'}</h3><p>{e(v['answer'])}</p><p><b>评审</b>　{e(j['verdict'])}　引用 {e(j['citation'])}</p><p>{e(j['why'])}</p><details><summary>展开生成器看到的五段原文（标注 Hit@5={int(v['hit5'])}）</summary><ol>{items}</ol></details></section>"
cats=['对→错','错→对','两组均错','两组均对','待复核']
buttons=''.join(f'<button data-group="{k}">{k} · {counts[k]}</button>' for k in cats)
page=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>SC500 答案配对复核</title><style>body{{font:16px/1.65 system-ui;max-width:1200px;margin:36px auto;padding:0 20px;color:#182940}}article{{border-top:2px solid #c4d0db;padding:15px 0}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}section{{background:#f2f6f8;padding:16px;min-width:0;overflow-wrap:anywhere}}summary,button{{cursor:pointer}}button,input{{padding:6px;margin:4px}}button.active{{background:#234c68;color:white}}li{{margin-bottom:9px}}@media(max-width:750px){{.pair{{grid-template-columns:1fr}}}}</style>
<h1>SC500 答案配对复核</h1><p>同题比较 Dense 与 Dense + DeepSeek 重排各自的前五段。两组均用项目的回答提示词调用 DeepSeek 生成中文答案。另一次 DeepSeek 调用依据同一参考答案、标注证据及各自的五段资料，在隐藏方案名称、按题交换展示顺序后评审。这里记录模型判定，未做独立人工逐题复核。</p>
<p>先选“对→错”检查损失，再选“错→对”检查收益。先读题目、参考答案和两份回答，再展开原文核对引用；correct 表示核心答案完整正确，incorrect 含实质错误、遗漏或拒答，uncertain 表示材料不足以裁决。引用单独评分，不作为正确性的硬门槛。数值、对象与范围优先于措辞是否相似。</p>
<button class="active" data-group="全部">全部 · 500</button>{buttons}<input id="query" placeholder="搜索 qid 或英文题目"><span id="shown"></span>"""
for r in sorted(rows,key=lambda r:(cats.index(transition(r)),r['qid'])):
    gold=''.join('<li>'+e(x)+'</li>' for x in r['gold_evidence'])
    page+=f"""<article data-group="{transition(r)}" data-search="{e(r['qid']+' '+r['question'].lower())}"><h2>{transition(r)} · {e(r['qid'])}</h2><p>{e(r['question'])}</p><details><summary>查看参考答案及标注段落</summary><p>{e(' / '.join(r['references']))}</p><ul>{gold}</ul></details><div class="pair">{answer(r,'dense')}{answer(r,'rerank')}</div></article>"""
page+="""<script>let group='全部';const query=document.querySelector('#query');function update(){let n=0;document.querySelectorAll('article').forEach(a=>{a.hidden=!(group==='全部'||a.dataset.group===group)||!a.dataset.search.includes(query.value.toLowerCase().trim());if(!a.hidden)n++});document.querySelector('#shown').textContent=n+' 题'};document.querySelectorAll('button').forEach(b=>b.onclick=()=>{group=b.dataset.group;document.querySelectorAll('button').forEach(v=>v.classList.toggle('active',v===b));update()});query.oninput=update;update()</script></html>"""
(out/'review.html').write_text(page)
