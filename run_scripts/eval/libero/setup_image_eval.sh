#!/bin/bash
# Isolate evaluation additions from the existing simulator and training envs.
set -euo pipefail
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"
BASE_PYTHON="${LIBERO_BASE_PYTHON:-/sjw_alinlab/home/hojin2/miniconda3/envs/libero310/bin/python}"
if [[ ! -x .venv-libero-eval/bin/python ]]; then
    uv --cache-dir .cache/uv venv .venv-libero-eval --python "$BASE_PYTHON" --system-site-packages
fi
uv --cache-dir .cache/uv pip install --python .venv-libero-eval/bin/python \
    numpy==1.26.4 gymnasium==0.29.1 pyzmq==27.2.0 msgpack==1.2.2 \
    imageio-ffmpeg==0.6.0 easydict==1.13 robosuite==1.4.0 mujoco==2.3.7 bddl==1.0.1
mkdir -p .cache/libero-eval
.venv-libero-eval/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('external_dependencies/LIBERO/libero/libero').resolve()
config = {'benchmark_root': str(root), 'bddl_files': str(root/'bddl_files'),
          'init_states': str(root/'init_files'), 'datasets': str(root.parent/'datasets'),
          'assets': str(root/'assets')}
Path('.cache/libero-eval/config.yaml').write_text(json.dumps(config, indent=2))
PY
