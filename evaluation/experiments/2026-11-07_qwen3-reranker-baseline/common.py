"""Frozen-cache evaluation. This module does not import a model or call an API."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def load_inputs():
    manifest = read_json(ROOT / "data/manifest.json")
    for filename, sha in manifest["files"].items():
        check(digest(ROOT / "data" / filename) == sha, f"数据校验失败: {filename}")
    queries = read_rows(ROOT / "data/queries.jsonl")
    docs = {r["id"]: r for r in read_rows(ROOT / "data/documents.jsonl")}
    check(len({q['qid'] for q in queries}) == len(queries), "重复 qid")
    for q in queries:
        check(len(q['candidates']) == len(set(q['candidates'])) == 50, f"候选必须是50个不同的块: {q['qid']}")
        check(set(q['candidates']) <= docs.keys(), f"缺失候选正文: {q['qid']}")
    return queries, docs


def rank_indices(scores):
    check(all(isinstance(s, (float, int)) and math.isfinite(s) for s in scores), "模型返回非有限分数")
    # Python's stable sort preserves Dense order when logit differences tie.
    return sorted(range(len(scores)), key=lambda i: -scores[i])


def metrics(ranked_ids, label):
    covered = [set(label['coverage'].get(cid, [])) for cid in ranked_ids]
    groups = [set(g) for g in label['groups'] if g]
    check(bool(groups), "缺失有效证据标注")
    out = {}
    for k in (1, 5, 10, 50):
        found = set().union(*covered[:k])
        recall = max(len(found & g) / len(g) for g in groups)
        out[f'hit@{k}'] = int(any(covered[:k]))
        out[f'recall@{k}'] = recall
        out[f'all_evidence@{k}'] = int(recall == 1)
    for k in (5, 10):
        out[f'mrr@{k}'] = next((1 / i for i, c in enumerate(covered[:k], 1) if c), 0.0)
    return out


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def validate_prediction(row, query):
    check(row['qid'] == query['qid'], '缓存 qid 不匹配')
    check(row['candidate_ids'] == query['candidates'], '缓存候选顺序不匹配')
    check(len(row['scores']) == 50, '缓存分数不足50个')
    expected = [query['candidates'][i] for i in rank_indices(row['scores'])]
    check(row['ranked_ids'] == expected, '缓存排序或分数无效')


def summarize(queries, predictions=None):
    # Gold labels enter here, after model scoring. Never passed to the model.
    labels = {r['qid']: r for r in read_rows(ROOT / 'data/labels.jsonl')}
    rows = []
    for q in queries:
        label = labels[q['qid']]
        ranked = q['candidates'] if predictions is None else predictions[q['qid']]['ranked_ids']
        row = {'qid': q['qid'], 'group': q['group'], 'metrics': metrics(ranked, label)}
        if predictions is not None:
            p = predictions[q['qid']]
            validate_prediction(p, q)
            row.update({k: p[k] for k in ('seconds', 'truncated_pairs', 'pair_count', 'peak_allocated_gib')})
        rows.append(row)
    summary = {}
    for group in sorted({q['group'] for q in queries}):
        subset = [r for r in rows if r['group'] == group]
        result = {'n': len(subset), 'metrics': {k: statistics.mean(r['metrics'][k] for r in subset) for k in subset[0]['metrics']}}
        if predictions is not None:
            seconds = [r['seconds'] for r in subset]
            result.update(p50_ms=1000 * statistics.median(seconds), p95_ms=1000 * quantile(seconds, .95),
                          pairs_per_second=sum(r['pair_count'] for r in subset) / sum(seconds),
                          truncated_pairs=sum(r['truncated_pairs'] for r in subset),
                          pair_count=sum(r['pair_count'] for r in subset),
                          peak_allocated_gib=max(r['peak_allocated_gib'] for r in subset))
        summary[group] = result
    return summary, rows
