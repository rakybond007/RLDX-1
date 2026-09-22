#!/bin/bash
#SBATCH --job-name=train_rldx1_pt_image_robocasa_h100_sharded_60k_globalbatch64_2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --partition=h100
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --time=2-00:00:00
#SBATCH --output=out/%j-rldx1_img_robocasa_h100.out
#SBATCH --error=out/%j-rldx1_img_robocasa_h100.err

set -euo pipefail
export NUM_WORKERS="${NUM_WORKERS:-32}"
export OUTPUT_DIR="${OUTPUT_DIR:-/rlwrld-unified-checkpoints/hojin2/checkpoints/rldx1_image_baselines}"
export RUN_NAME=rldx1_img_robocasa_gb64_60k_baseline
export MAX_STEPS=60000
export GRAD_ACCUM=1
export TRAIN_ENTRYPOINT=rldx/experiment/launch_train.py
printf 'H100 training: GPUs=2 CPUs=partition-default workers/rank=%s\n' "$NUM_WORKERS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi topo -m
exec bash run_scripts/train/benchmarks/finetune_rldx1_robocasa_image_2gpu.sh
