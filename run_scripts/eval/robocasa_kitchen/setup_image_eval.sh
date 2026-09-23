#!/bin/bash
# Prepare an isolated RoboCasa client with writable copies of upstream assets.
set -euo pipefail
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"
BASE_PYTHON="${ROBOCASA_BASE_PYTHON:-/sjw_alinlab/home/hojin2/miniconda3/envs/robocasa_gr00t/bin/python}"
ROBOCASA_ASSET_SOURCE="${ROBOCASA_ASSET_SOURCE:-/sjw_alinlab/home/hojin2/multigpu_workspace/robocasa/robocasa/models/assets}"
ROBOSUITE_SOURCE="${ROBOSUITE_SOURCE:-/sjw_alinlab/home/hojin2/multigpu_workspace/robosuite/robosuite}"
test -d "$ROBOCASA_ASSET_SOURCE/objects/objaverse"
test -f "$ROBOSUITE_SOURCE/macros.py"
if [[ ! -x .venv-robocasa-eval/bin/python ]]; then
    uv --cache-dir .cache/uv venv .venv-robocasa-eval --python "$BASE_PYTHON" --system-site-packages
fi
uv --cache-dir .cache/uv pip install --python .venv-robocasa-eval/bin/python \
    numpy==1.26.4 gymnasium==0.29.1 msgpack-numpy==0.4.8
mkdir -p .work/robocasa_local/robocasa .work/robosuite_local/robosuite
rsync -a --exclude 'models/assets/' --exclude '__pycache__/' \
    external_dependencies/robocasa/robocasa/ .work/robocasa_local/robocasa/
rsync -a --exclude 'models/assets/' --exclude '__pycache__/' \
    "$ROBOSUITE_SOURCE/" .work/robosuite_local/robosuite/
python3 - <<'PY'
from pathlib import Path
p=Path('.work/robosuite_local/robosuite/macros.py')
s=p.read_text()
assert s.count('CACHE_NUMBA = True') == 1 or s.count('CACHE_NUMBA = False') == 1
p.write_text(s.replace('CACHE_NUMBA = True', 'CACHE_NUMBA = False'))
PY
ln -sfn "$ROBOSUITE_SOURCE/models/assets" .work/robosuite_local/robosuite/models/assets
# The local patch writes temporary XML to /tmp and resolves mesh paths to
# their original absolute locations, so source assets can stay read-only.
ln -sfn "$ROBOCASA_ASSET_SOURCE" .work/robocasa_local/robocasa/models/assets
python3 run_scripts/eval/robocasa_kitchen/patch_readonly_assets.py \
    .work/robocasa_local/robocasa/models/objects/objects.py
echo 'RoboCasa image evaluation environment and shared assets are ready.'
