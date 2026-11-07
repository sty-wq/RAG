"""Build the human-readable conclusion from completed P0-P2 results."""
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
s=json.loads((OUT/'original1309/summary.json').read_text())
t=json.loads((OUT/'rewritten500/summary.json').read_text())
v=json.loads((OUT/'verification.json').read_text())
assert s['n']==1309 and t['n']==500
assert v['real_candidate_fusions_match_production']==1809
assert s['provenance']['analysis_sha256']==t['provenance']['analysis_sha256']==v['analysis_sha256']
assert s['historical_transitions_rechecked']['regression_qid_set_identical']
assert s['historical_transitions_rechecked']['gain_qid_set_identical']
rows={r['qid']:r for r in map(json.loads,(OUT/'original1309/details.jsonl').read_text().splitlines())}
chunks={c['id']:c for c in map(json.loads,(OUT/'chunks.jsonl').read_text().splitlines())}
labels={'dense':'Dense / α=0','bm25':'BM25','rrf_a0.25':'RRF，BM25 权重 0.25',
        'rrf_a0.5':'RRF，BM25 权重 0.5','rrf_a1':'当前等权 RRF，权重 1'}
lines=['# P0–P2 混合检索退化分析','', '日期：2026-11-04。项目：PythonProject5。','',
    '**结论：在这批 Qasper 题上，BM25 的前几名段落排序显著弱于 Dense。当前等权 RRF 将这一路较弱的排序引入最终结果，导致正确证据从前五后退。历史 214 道 Hit@5 退化题全部仍在融合前 50 条中找到证据，主要损失发生在排序阶段。**','',
    '本轮已完成 P0 复现与输入冻结、P1 候选互补性与丢失位置分析、P2 固定候选的权重消融。原始代码、索引、历史结果未修改，没有重排模型、回答模型或 LLM 评分调用。补齐原题查询向量使用了 embedding API。','',
    '## 1. 实验控制与复现边界','',
    '- 主实验为原 1309 道可计分题，语料 416 篇论文、17887 块。原问题包含 1294 道 title 组和 15 道 self 组。',
    '- 冻结问题文本、证据、语料 SHA256、索引实际数据文件、源码以及历史 A/B/C 明细。采用索引副本，保留每题查询向量和两路各 50 候选的 ID、名次、分数及可回查原文。',
    '- 查询全库，不用 gold_docs 做检索过滤。沿用原 BM25、原标题加正文输入、原分词、原切块、原证据匹配规则。',
    '- RRF 常数固定为 60。只改变 BM25 权重，Dense 权重固定为 1。α=0 严格返回原 Dense 顺序。',
    '- 原题旧结果只保存前10，因此本轮补算了查询向量和 Dense50。新旧 API 查询向量没有同时保留，少数排序变化不能直接归因为浮点误差、模型切换或索引近似搜索中的任何单一因素。','',
    '| P0 核验 | 结果 |','|---|---|',
    f"| BM25 前10顺序与分数 | 全部 1309 题与历史一致 |",
    f"| Dense 前10顺序 | {s['p0']['A_top10_identical']}/1309 题与历史一致 |",
    f"| Hybrid 前10顺序 | {s['p0']['C_top10_identical']}/1309 题与历史一致 |",
    f"| 本轮 Dense k=10 与 k=50 前缀 | {s['p0']['dense_k10_vs_k50_prefix_identical']}/1309 题一致 |",
    '| Hit@1、Hit@5 | 三组均复现历史聚合值 |',
    '| Hit@5 变好、变差名单 | 70 道改善、214 道退化，qid 集合完全一致 |','',
    '仍有两处指标差异必须保留：Dense 的 1909.10481-q03 首个正确块从第7变第8，使平均 MRR 从 0.40832242 变为 0.40830878；Hybrid 的 1901.09501-q02 Recall@5 提高 0.5，使平均 Recall@5 从历史 0.43983975 变为本轮 0.44022172。不能声称逐条完全复现。下表使用本轮固定候选结果。','',
    f'[新旧顺序与指标差异]({OUT}/original1309/reproduction-differences.json)','',
    '## 2. P2 主实验结果','',
    '| 方法 | Hit@1 | Hit@5 | Recall@5 | Recall@10 | MRR@10 |','|---|---|---|---|---|---|']
for arm,label in labels.items():
    m=s['metrics'][arm]
    lines.append('| '+label+' | '+' | '.join(f'{m[k]:.4f}' for k in ('hit@1','hit@5','recall@5','recall@10','mrr@10'))+' |')
lines += ['',
    '**降低 BM25 权重能缓解退化，但本轮所有非零权重仍未超过 Dense。不能把 α=0.25 当成已验证优于 Dense 的新默认方案。**','',
    '| BM25 权重 | 恢复原214道损失 | 保留原70道收益 | 相对Dense新增命中 | 相对Dense丢失命中 |',
    '|---|---|---|---|---|']
for arm in ('rrf_a0','rrf_a0.25','rrf_a0.5','rrf_a1'):
    e=s['p2'][arm]
    lines.append(f"| {arm.removeprefix('rrf_a')} | {int(e['restored_current_hybrid_losses'])}/214 | {int(e['retained_current_hybrid_gains'])}/70 | {e['vs_dense_hit5_improved']} | {e['vs_dense_hit5_regressed']} |")
lines += ['',
    'α=0.25 恢复 154/214 道历史损失，但只保留 28/70 道历史收益，并产生部分新的退化；最终仍比 Dense 少命中31题。α=0 恢复全部 Dense 结果，也不再保留相对 Dense 的新增收益。','',
    '## 3. P1 正确证据究竟去了哪里','',
    '| 对历史214道退化题的检查 | 题数 |','|---|---|',
    '| 正确证据仍在融合前50 | 214 |',
    '| 融合前50完全找不到标注证据 | 0 |',
    '| Dense最靠前正确块同时进入BM25前50 | 210 |',
    '| Dense最靠前正确块只在Dense前50 | 4 |',
    '| 融合前5仍命中正确论文 | 214 |','',
    '这些题的最靠前正确块，Dense 名次中位数是 **3**，融合后名次中位数是 **9**。在两路都召回该块的210题中，它的 BM25 名次中位数是 **20**。','',
    '**因此需要修正最初的假设：RRF 强奖励共同候选确实存在，但“正确块仅由Dense召回而被压后”只覆盖4/214题。更主要的是两路都召回正确块以后，BM25 将它排得较低，等权融合改变了同一批候选的前列顺序。**','',
    'P0 与函数一致性验证没有发现 ID 映射、排序方向或 RRF 公式实现偏离当前设计。问题体现为当前排序信号与融合策略不适合这批题，而不是本轮发现了程序算错。','',
    '## 4. 一个可直接复查的样本','']
r=rows['1905.00840-q03'];cid=r['drift']['first_dense_gold'];h=chunks[cid]
lines += [f"题号 `{r['qid']}`",'',r['question'],'',
    '问题问 KALM 使用什么评测数据集。正确块位于 Evaluations，明确说明手工构造的问题集和 MetaQA。','',
    '| 候选内容 | Dense名次 | BM25名次 | 等权融合名次 |','|---|---|---|---|',
    '| Evaluations，包含评测数据集 | 1 | 12 | 7 |',
    '| Related Works，说明支持知识编写与问答 | 4 | 1 | 1 |','',
    '正确块的融合分数为 `1/61 + 1/72 ≈ 0.030282`；Related Works 的融合分数为 `1/64 + 1/61 ≈ 0.032018`。后者没有回答“用了什么数据集”，却因 BM25 排第一而超过正确块。','',
    '正确块原文','', '> '+h['text'].replace('\n','\n> '),'',
    f'正确块 ID：`{cid}`。各候选全文及权重后的排名都保存在逐题明细。','',
    '## 5. BM25 有没有提供额外召回价值','',
    '| 候选集 | Hit@50 / 并集Hit | Recall@50 / 并集Recall | 完整覆盖题数 |','|---|---|---|---|']
for arm,label in [('dense','Dense50'),('bm25','BM25 50'),('rrf_a1','当前融合50')]:
    m=s['metrics'][arm]
    lines.append(f"| {label} | {m['hit@50']:.4f} | {m['recall@50']:.4f} | {round(m['full_recall@50']*1309)}/1309 |")
p=s['p1']
lines.append(f"| 两路并集（最多100，非同预算） | {p['union_hit']:.4f} | {p['union_recall']:.4f} | {p['union_full_recall_questions']}/1309 |")
lines += ['',
    f"BM25 有独有证据单元的题共 {p['questions_with_bm25_unique_evidence']} 道；并集相对 Dense50 实际提高历史 Recall 口径的有 {p['union_recall_gt_dense50_questions']} 道。两者不同，因为 Recall 取多个标注组中覆盖最好的一组。",'',
    f"融合截断到50，相对并集使 {p['fusion_recall50_lt_union_questions']} 题 Recall 下降；相对 Dense50，使 {p['fusion_recall50_lt_dense50_questions']} 题 Recall 下降、{p['fusion_hit50_lt_dense50_questions']} 题 Hit@50 下降。但总体融合50的平均 Recall 仍略高于 Dense50，所以不能把前五退化解释成整体候选召回都变差。",'',
    '这说明 BM25 有补充候选的价值，但不代表应让它与 Dense 等权决定最终前几名。','',
    '## 6. RRF 的两条数学约束','',
    '1. 当每路50、常数60、等权时，最差共同候选为 `2/110≈0.01818`，最好独有候选为 `1/61≈0.01639`。所有共同候选都排在所有独有候选之前。本轮原1309题的融合前五全部来自交集。',
    '2. 当 `α < 61/110≈0.5545` 时，BM25 独有候选的最高分仍小于 Dense 第50名的最低分。因此本轮 α=.25/.5 的融合前50集合等于 Dense50，只改变其内部排序；并没有纳入 BM25 独有候选。','',
    '这也解释了为什么单纯减小权重会在“保护Dense排序”和“使用BM25独有证据”之间产生冲突。若下一步测试候选配额/并集合并，应作为新的消融配置，不混入本轮。','',
    '## 7. 改写500题的补充回放','',
    '使用已经存在的500题 Dense50 缓存，没有重新生成查询向量。这是额外诊断，不是独立留出集上的优化验收。','',
    '| 方法 | Hit@5 | Recall@5 | MRR@10 |','|---|---|---|---|']
for arm,label in labels.items():
    m=t['metrics'][arm]
    lines.append('| '+label+' | '+' | '.join(f'{m[k]:.4f}' for k in ('hit@5','recall@5','mrr@10'))+' |')
lines += ['',
    '等权融合相对 Dense 变好31题、变差68题。68题全部仍在融合前50找到证据，其中59题最靠前正确块两路都召回、9题仅Dense召回。α=.25恢复42/68道损失，但只保留12/31道原收益，仍低于 Dense。', '',
    '退化现象在改写问题上依然存在，不能只归咎于旧题带论文标题。标题词、索引标题重复、分词和长度归一化的贡献仍需 P3 控制变量实验。','',
    '## 8. 本轮结论与下一步','',
    '- 已证实：前列排序质量是主要退化位置；BM25 与 Dense 等权融合会在这两组题上损害 Hit@5、Recall@5 和 MRR。',
    '- 已证实：BM25 仍有少量独有证据；候选召回质量和最终前五排序质量必须分开评估。',
    '- 已证实：减小 BM25 权重能恢复不少损失，但测试的非零权重均未超过 Dense，尚不能作为新的更优默认配置。',
    '- 尚未证实：BM25 较差排序究竟由标题、词项处理、长度归一化或任务本身的语义需求分别贡献多少。下一步按原计划做 P3，固定 Dense 候选，对 BM25 查询是否含标题、索引是否含论文标题做2×2实验。',
    '- 生产代码未修改。权重属于本轮诊断参数；任何最终策略都要在冻结的独立测试集验收。',
    '- 以上为证据检索指标，不是问答准确率。长段落的足够长片段也可能被历史 covers 规则计为覆盖。','',
    '## 9. 产物与验证','',
    f'- [主实验完整报告]({OUT}/original1309/报告.md)',
    f'- [主实验逐题明细]({OUT}/original1309/details.jsonl)',
    f'- [代表性错误与互补样本]({OUT}/original1309/cases.md)',
    f'- [500题补充报告]({OUT}/rewritten500/报告.md)',
    f'- [冻结清单]({OUT}/manifest.json)',
    f'- [执行与阅读方法]({OUT}/README.md)',
    f'- [验证记录]({OUT}/verification.json)','',
    '已检查1809题的等权融合输出与生产 fuse_rrf 完全一致、α=0与Dense顺序完全一致；8道不同问题的完整BM25前50与生产 lexical_search（控制存储迭代器）一致；边界与同分情况5项断言通过。原1309题BM25前10顺序和分数与真实历史结果完全一致。查询索引副本的实际搜索也已验证。','']
path=OUT/'P0-P2-混合检索退化分析.md'
path.write_text('\n'.join(lines))
print(path)
