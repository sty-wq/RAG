#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
exec /Users/bia/PycharmProjects/PythonProject5/.venv/bin/python experiment.py "$@"
