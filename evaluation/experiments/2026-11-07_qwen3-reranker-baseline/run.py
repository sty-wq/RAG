"""Run one fixed Qwen baseline; checkpoint after every question."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from collections import Counter
from pathlib import Path

from common import ROOT, check, digest, load_inputs, rank_indices, read_json, summarize, validate_prediction, write_json
from score import INSTRUCTION, MODELS, QwenScorer


def select(queries, limit_per_group):
    seen = Counter()
    selected = []
    for q in queries:
        if not limit_per_group or seen[q['group']] < limit_per_group:
            selected.append(q)
            seen[q['group']] += 1
    return selected


def document_text(doc):
    return (doc['heading_path'] + '\n' + doc['text']).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=list(MODELS))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--limit-per-group', type=int, default=0, help='0=全部；2=每组2题，共6题')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-length', type=int, default=2048)
    args = parser.parse_args()
    check(args.batch_size > 0 and args.max_length > 0 and args.limit_per_group >= 0, '参数必须为有效正数')
    import torch
    check(torch.cuda.is_available(), 'CUDA 不可用')
    queries, docs = load_inputs()
    queries = select(queries, args.limit_per_group)
    gpu = torch.cuda.get_device_properties(0)
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'tokenizers', 'huggingface-hub', 'safetensors')}
    config = {'model': MODELS[args.model][0], 'revision': MODELS[args.model][1],
              'data_sha256': digest(ROOT / 'data/manifest.json'),
              'code_sha256': {n: digest(ROOT / n) for n in ('run.py', 'score.py', 'common.py')},
              'qids': [q['qid'] for q in queries], 'batch_size': args.batch_size,
              'max_length': args.max_length, 'dtype': 'bfloat16', 'attention': 'sdpa',
              'instruction': INSTRUCTION, 'document_format': 'heading_path + newline + text',
              'versions': versions, 'python': platform.python_version(), 'cuda': torch.version.cuda,
              'gpu': gpu.name, 'gpu_memory_bytes': gpu.total_memory,
              'score': 'logit_yes - logit_no; descending; Dense order breaks ties'}
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / 'config.json'
    if config_path.exists():
        check(read_json(config_path) == config, '输出目录的模型、参数、数据或环境已变化；请使用新目录')
    else:
        check(not list(args.output.iterdir()), '输出目录不为空且缺少配置，停止以免混用结果')
        write_json(config_path, config)
    predictions = {}
    for q in queries:
        file = args.output / 'items' / (q['qid'] + '.json')
        if file.exists():
            prediction = read_json(file)
            validate_prediction(prediction, q)
            predictions[q['qid']] = prediction
    pending = [q for q in queries if q['qid'] not in predictions]
    print(f"{args.model}: {len(queries)}题，已完成{len(predictions)}题，待评分{len(pending)}题", flush=True)
    if pending:
        torch.manual_seed(0)
        print('加载固定版本模型（首次需要下载权重）…', flush=True)
        started = time.perf_counter()
        scorer = QwenScorer(args.model, args.max_length, args.batch_size)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - started
        sanity_scores, _ = scorer.score('What is the capital of China?', ['The capital of China is Beijing.', 'Gravity attracts masses.'])
        check(sanity_scores[0] > sanity_scores[1], '基本相关性检查失败，停止运行')
        warm = pending[0]
        for _ in range(2):
            scorer.score(warm['question'], [document_text(docs[c]) for c in warm['candidates']])
        torch.cuda.synchronize()
        session_id = time.time_ns()
        write_json(args.output / 'sessions' / f'{session_id}.json',
                   {'model_load_seconds_including_download': load_seconds, 'sanity_scores': sanity_scores,
                    'warmup_queries': 2, 'parameter_count': sum(p.numel() for p in scorer.model.parameters()),
                    'pending_qids': [q['qid'] for q in pending]})
        for q in pending:
            documents = [document_text(docs[c]) for c in q['candidates']]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            scores, lengths = scorer.score(q['question'], documents)
            ranked_ids = [q['candidates'][i] for i in rank_indices(scores)]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            prediction = {'qid': q['qid'], 'candidate_ids': q['candidates'], 'scores': scores,
                          'ranked_ids': ranked_ids, 'seconds': elapsed, 'session': session_id,
                          'pair_count': len(scores), 'raw_token_lengths': lengths,
                          'truncated_pairs': sum(n > args.max_length for n in lengths),
                          'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3}
            validate_prediction(prediction, q)
            write_json(args.output / 'items' / (q['qid'] + '.json'), prediction)
            predictions[q['qid']] = prediction
            print(f"{len(predictions)}/{len(queries)} {q['qid']} {elapsed:.3f}s 截断{prediction['truncated_pairs']}/50", flush=True)
    summary, details = summarize(queries, predictions)
    write_json(args.output / 'summary.json', summary)
    write_json(args.output / 'details.json', details)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
