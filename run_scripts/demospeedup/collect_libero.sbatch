#!/bin/bash
#SBATCH --job-name=collect_rldx1_image_libero60000_demospeedup_slow2_fast4_entropy
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=sjw_alinlab_premium
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --array=0-31%2
#SBATCH --time=1-00:00:00
#SBATCH --output=out/%A_%a-libero_entropy.out
#SBATCH --error=out/%A_%a-libero_entropy.err
set -euo pipefail
export PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HOME="$PWD/.cache/huggingface"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CKPT="${CKPT:-/rlwrld-unified-checkpoints/hojin2/checkpoints/rldx1_image_baselines/rldx1_img_libero_gb32_60k_baseline/checkpoint-60000}"
DATASET="${DATASET:-/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta}"
.venv/bin/python run_scripts/demospeedup/collect_entropy.py \
    --model-path "$CKPT" --dataset-path "$DATASET" \
    --output "$PWD/outputs/demospeedup/libero_60000_slow2_fast4" \
    --batch-size 8 --n-samples 10 --slow 2 --fast 4 \
    --num-shards 32 --shard-index "${SLURM_ARRAY_TASK_ID:-0}"
