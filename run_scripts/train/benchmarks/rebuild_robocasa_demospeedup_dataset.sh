#!/bin/bash
#SBATCH --job-name=rebuild_robocasa_rldx1_demospeedup_slow2_fast4_full_7200_episodes
#SBATCH --partition=cpu
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --time=12:00:00
#SBATCH --output=out/%j-robocasa_demospeedup_rebuild.out
#SBATCH --error=out/%j-robocasa_demospeedup_rebuild.err

set -euo pipefail
cd /sjw_alinlab/home/hojin2/quantization_agent_workspace/RLDX-1
SOURCE="${SOURCE:-/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300}"
KIT="${KIT:-$PWD/.work/robocasa-entropy-package/robocasa-entropy}"
OUT="${OUT:-$PWD/.work/datasets/robocasa_demospeedup_slow2_fast4}"
[[ ! -e "$OUT" ]] || { echo "Output exists; refusing to mix datasets: $OUT" >&2; exit 1; }
"$PWD/.venv/bin/python" run_scripts/train/benchmarks/rebuild_robocasa_demospeedup_dataset.py \
    --source "$SOURCE" --kit "$KIT" --out "$OUT"
