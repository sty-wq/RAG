#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -c 'import torch; assert torch.cuda.is_available(); print("Using existing torch:", torch.__version__)'
# Inherit the verified CUDA build, but keep our Python dependencies isolated.
python -m venv --system-site-packages .venv
# A nested venv otherwise loses packages installed in the active py312 venv.
# Add that environment's site-packages after the new environment's own packages.
python - <<'PY'
import json
import pathlib
import site
import subprocess
target = subprocess.check_output(['.venv/bin/python', '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip()
pathlib.Path(target, 'zz_baseline_parent.pth').write_text('\n'.join(site.getsitepackages()) + '\n')
PY
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'import torch; from transformers import Qwen3ForCausalLM; assert torch.cuda.is_available(); print("依赖导入和GPU检查通过", torch.__version__)'
.venv/bin/python -m unittest -v test_baseline.py
.venv/bin/python verify.py
