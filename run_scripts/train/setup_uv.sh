#!/bin/bash
# Run from the repository root, preferably on the allocated CUDA node.
set -euo pipefail
[[ -f pyproject.toml && -f uv.lock ]] || { echo 'Run from the RLDX-1 root.' >&2; exit 1; }
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
export MAX_JOBS="${MAX_JOBS:-8}"
# Frozen avoids re-resolving optional TensorRT build metadata during training setup.
uv sync --python 3.10 --frozen
.venv/bin/python -c 'import torch, flash_attn, rldx; print("torch", torch.__version__, "flash_attn", flash_attn.__version__, "rldx", rldx.__version__); print("CUDA available:", torch.cuda.is_available(), "GPU count:", torch.cuda.device_count())'
