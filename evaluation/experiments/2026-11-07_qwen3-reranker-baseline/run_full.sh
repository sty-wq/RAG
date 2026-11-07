#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
.venv/bin/python -u run.py --model 0.6b --output results/full-06b
.venv/bin/python -u run.py --model 4b --output results/full-4b
.venv/bin/python compare.py results/full-06b results/full-4b --output results/comparison
