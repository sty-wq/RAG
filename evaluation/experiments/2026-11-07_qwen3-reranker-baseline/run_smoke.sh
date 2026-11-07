#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
.venv/bin/python -u run.py --model 0.6b --limit-per-group 2 --output results/smoke-06b
.venv/bin/python -u run.py --model 4b --limit-per-group 2 --output results/smoke-4b
.venv/bin/python compare.py results/smoke-06b results/smoke-4b --output results/smoke-comparison
