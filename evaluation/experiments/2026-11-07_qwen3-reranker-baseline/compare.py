"""Compare Dense and completed model runs on exactly the same queries."""
import argparse
import json
from pathlib import Path

from common import check, digest, load_inputs, read_json, summarize, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results/comparison'))
    args = parser.parse_args()
    all_queries, _ = load_inputs()
    configs = [read_json(p / 'config.json') for p in args.runs]
    for config in configs[1:]:
        for key in configs[0]:
            if key not in ('model', 'revision'):
                check(config[key] == configs[0][key], f'两组设置不一致: {key}')
    from common import ROOT
    check(configs[0]['data_sha256'] == digest(ROOT / 'data/manifest.json'), '当前数据与运行配置不一致')
    selected_ids = set(configs[0]['qids'])
    queries = [q for q in all_queries if q['qid'] in selected_ids]
    check([q['qid'] for q in queries] == configs[0]['qids'], '题目列表不一致')
    dense_summary, dense_details = summarize(queries)
    results = {'Dense': dense_summary}
    baseline = {r['qid']: r for r in dense_details}
    changes = {}
    for folder, config in zip(args.runs, configs):
        predictions = {q['qid']: read_json(folder / 'items' / (q['qid'] + '.json')) for q in queries}
        summary, details = summarize(queries, predictions)
        name = config['model'].split('/')[-1]
        check(name not in results, '模型重复，无法作为两个不同版本比较')
        results[name] = summary
        changes[name] = []
        for row in details:
            changes[name].append({'qid': row['qid'], 'group': row['group'],
                                  'dense': baseline[row['qid']]['metrics'], 'rerank': row['metrics'],
                                  'delta': {k: v - baseline[row['qid']]['metrics'][k] for k, v in row['metrics'].items()}})
    text = ['# Qwen3-Reranker 基准对照', '',
            '同一题目、同一 Dense Top50 候选。这里只评估检索排序；没有生成答案，也没有计算回答置信度。', '',
            'P50/P95 为预热后每题50个候选的分词、GPU评分及排序耗时，不含 embedding、模型加载和答案生成。', '']
    columns = ['hit@1', 'hit@5', 'recall@5', 'mrr@5', 'all_evidence@5', 'recall@50']
    for group in dense_summary:
        text.extend([f"## {group}（{dense_summary[group]['n']}题）", '',
                     '| 方法 | Hit@1 | Hit@5 | Recall@5 | MRR@5 | 证据找齐@5 | Recall@50 | P50(ms) | P95(ms) |',
                     '|---|---|---|---|---|---|---|---|---|'])
        for name, summary in results.items():
            r = summary[group]
            values = [f"{r['metrics'][k]:.4f}" for k in columns]
            values += [f"{r[k]:.1f}" if k in r else '—' for k in ('p50_ms', 'p95_ms')]
            text.append('| ' + ' | '.join([name] + values) + ' |')
        text.append('')
    text.extend(['重排不会改变候选池，因此 Recall@50 应保持不变。证据找齐@5 是 Recall@5=1 的题目比例，不是 LLM 回答正确率。', '',
                 '单论文使用历史 Qasper 标注组的最优召回口径；跨论文使用每篇论文的必要事实引文匹配，不能把两种口径混作一个总分。', '',
                 '已有题目曾被分析，且93道跨论文题经过全文回答筛选。结果用于开发对比；后续微调需按论文划分并新增独立测试集。'])
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'summary.json', results)
    write_json(args.output / 'per_question_changes.json', changes)
    report = '\n'.join(text) + '\n'
    (args.output / 'report.md').write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
