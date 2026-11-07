#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
: "${EMBED_API_KEY:?请在已配置 QuickRouter EMBED_API_KEY 的终端运行}"
export EMBED_MODEL=text-embedding-3-small
export EMBED_BASE_URL=https://api.quickrouter.ai/v1
py=/Users/bia/PycharmProjects/PythonProject5/.venv/bin/python
"$py" evaluate.py
"$py" report.py
