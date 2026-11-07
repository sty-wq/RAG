#!/usr/bin/env bash
set -euo pipefail
cd /Users/bia/PycharmProjects/PythonProject5
: "${EMBED_API_KEY:?请在此前成功运行 embedding 的终端执行，需已配置 EMBED_API_KEY}"
export EMBED_MODEL='text-embedding-3-small'
export EMBED_BASE_URL='https://api.quickrouter.ai/v1'
export RERANK_MODEL='qwen3-rerank'
export RERANK_BASE_URL="$EMBED_BASE_URL"
export RERANK_API_KEY="$EMBED_API_KEY"
export RERANK_INSTRUCT='Given a web search query, retrieve relevant passages that answer the query.'
unset RAG_MIN_RERANK_SCORE
.venv/bin/python -m evaluation.run_retrieval \
  --data evaluation/data/qasper \
  --questions /Users/bia/Documents/Codex/2026-11-02/wo/outputs/qasper-pilot100/questions_ready_68.jsonl \
  --reuse-index --mode dense --candidate-k 50 --k-max 10 \
  --rerank http --rerank-query retrieval --label SC68_G_qwen3_rerank_v1
.venv/bin/python /Users/bia/Documents/Codex/2026-11-02/wo/outputs/compare_retrieval.py \
  --baseline evaluation/results/20261102-210812_20261102_210742_SC68_A_dense.json \
  evaluation/results/20261102-210917_20261102_210742_SC68_D_dense_rerank.json \
  evaluation/results/20261102-215135_SC68_F_deepseek_v2.json \
  evaluation/results/*_SC68_G_qwen3_rerank_v1.json
