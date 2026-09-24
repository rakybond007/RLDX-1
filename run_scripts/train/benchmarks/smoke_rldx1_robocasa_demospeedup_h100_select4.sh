#!/bin/bash
#SBATCH --job-name=smoke_rldx1_image_robocasa_demospeedup_h100_select4_healthy_gb64_micro16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --partition=h100
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --time=00:30:00
#SBATCH --output=out/%j-rldx1_robocasa_demospeedup_h100_select4_smoke.out
#SBATCH --error=out/%j-rldx1_robocasa_demospeedup_h100_select4_smoke.err

set -euo pipefail
cd /sjw_alinlab/home/hojin2/quantization_agent_workspace/RLDX-1
export DATA_DIR="$PWD/.work/datasets/robocasa_demospeedup_smoke32"
export NUM_GPUS=4
export MAX_STEPS=10
export SAVE_STEPS=1000
export NUM_WORKERS=8
export GRAD_ACCUM=1
export OUTPUT_DIR="$PWD/.work/smoke_checkpoints"
export RUN_NAME="robocasa_demospeedup_smoke32_h100_4gpu_gb64_${SLURM_JOB_ID}"
export RLDX_EARLY_CUDA_CHECK=1
export NCCL_DEBUG=WARN
export NCCL_CUMEM_HOST_ENABLE=0
echo "H100 smoke: job=$SLURM_JOB_ID reserved=8 active=$NUM_GPUS global_batch=64 microbatch=16"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
HEALTHY=()
for gpu in 0 1 2 3 4 5 6 7; do
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
