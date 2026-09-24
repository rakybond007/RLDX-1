#!/bin/bash
#SBATCH --job-name=train_rldx1_image_robocasa_demospeedup_slow2_fast4_h100_4active_5reserved_gb64_60k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=5
#SBATCH --partition=h100
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --time=2-00:00:00
#SBATCH --output=out/%j-rldx1_robocasa_demospeedup_h100_4gpu_train.out
#SBATCH --error=out/%j-rldx1_robocasa_demospeedup_h100_4gpu_train.err

set -euo pipefail
cd /sjw_alinlab/home/hojin2/quantization_agent_workspace/RLDX-1
export DATA_DIR="$PWD/.work/datasets/robocasa_demospeedup_slow2_fast4"
[[ -f "$DATA_DIR/meta/COMPLETE.json" ]] || { echo "Dataset is incomplete: $DATA_DIR" >&2; exit 1; }
export NUM_GPUS=4
export MAX_STEPS=60000
export SAVE_STEPS=2000
export NUM_WORKERS=10
export GRAD_ACCUM=1
export OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/checkpoints/rldx1_image_baselines
export RUN_NAME=rldx1_img_robocasa_demospeedup_slow2_fast4_gb64_60k_h100_4gpu
export RLDX_EARLY_CUDA_CHECK=1
export NCCL_DEBUG=WARN
export NCCL_CUMEM_HOST_ENABLE=0
echo "H100 training: job=$SLURM_JOB_ID reserved=5 active=$NUM_GPUS global_batch=64 microbatch=16 workers/rank=$NUM_WORKERS"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
HEALTHY=()
for gpu in 0 1 2 3 4; do
    if CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -c 'import torch; torch.cuda.set_device(0); torch.ones(1,device="cuda:0"); torch.cuda.synchronize()' >/dev/null 2>&1; then
        HEALTHY+=("$gpu")
        echo "CUDA_DEVICE_OK gpu=$gpu"
    else
        echo "CUDA_DEVICE_FAILED gpu=$gpu"
    fi
    if (( ${#HEALTHY[@]} == 4 )); then break; fi
done
(( ${#HEALTHY[@]} == 4 )) || { echo 'Fewer than four usable H100 GPUs'; exit 1; }
export CUDA_VISIBLE_DEVICES="${HEALTHY[0]},${HEALTHY[1]},${HEALTHY[2]},${HEALTHY[3]}"
echo "ACTIVE_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
.venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    run_scripts/train/benchmarks/probe_h100_4gpu.py
exec bash run_scripts/train/benchmarks/finetune_rldx1_robocasa_demospeedup_image_2gpu.sh
