#!/bin/bash
#SBATCH --job-name=rebuild_libero_demospeedup_slow2_fast4_verified_dataset
#SBATCH --partition=cpu
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --time=1-00:00:00
#SBATCH --output=out/%j-libero_demospeedup_rebuild.out
#SBATCH --error=out/%j-libero_demospeedup_rebuild.err

set -euo pipefail
BASE_DIR="${RLDX_ROOT:-$(pwd)}"
cd "$BASE_DIR"
SOURCE="${SOURCE:-/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta}"
PACKAGE="${PACKAGE:-$BASE_DIR/.work/libero-entropy-package/libero-entropy}"
OUT="${OUT:-$BASE_DIR/.work/datasets/libero_demospeedup_slow2_fast4}"
"$BASE_DIR/.venv/bin/python" "$PACKAGE/code/rebuild.py" --source "$SOURCE" --out "$OUT"
