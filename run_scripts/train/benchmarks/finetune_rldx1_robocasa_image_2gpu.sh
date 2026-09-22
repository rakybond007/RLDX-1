#!/bin/bash
#SBATCH --job-name=train_rldx1_pt_image_robocasa_baseline_60k_globalbatch64_2gpu
#SBATCH --comment="RLDX-1 image baseline, current-frame input, action chunk 16, no gradient accumulation"
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab_premium
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --time=2-00:00:00
#SBATCH --output=out/%j-rldx1_img_robocasa.out
#SBATCH --error=out/%j-rldx1_img_robocasa.err

# From repo root: mkdir -p out; sbatch run_scripts/train/benchmarks/finetune_rldx1_robocasa_image_2gpu.sh
# In an existing 2-GPU srun shell: MAX_STEPS=10 SAVE_STEPS=10 bash <this script>
set -euo pipefail
BASE_DIR="${RLDX_ROOT:-$(pwd)}"
if [[ ! -f "$BASE_DIR/rldx/experiment/launch_train.py" ]]; then
    echo 'Run from the RLDX-1 repository root, or set RLDX_ROOT.' >&2
    exit 1
fi
cd "$BASE_DIR"
DATA_DIR="${DATA_DIR:-/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300}"
if [[ -z "${BASE_MODEL_PATH:-}" ]]; then
    if [[ -f "$BASE_DIR/models/RLDX-1-PT-IMG/model.safetensors.index.json" ]]; then
        BASE_MODEL_PATH="$BASE_DIR/models/RLDX-1-PT-IMG"
    else
        BASE_MODEL_PATH="RLWRLD/RLDX-1-PT-IMG"
    fi
fi
NUM_GPUS=2
GLOBAL_BATCH_SIZE=64
GRAD_ACCUM="${GRAD_ACCUM:-1}"
if [[ "$GRAD_ACCUM" != 1 ]]; then
    echo 'This baseline requires GRAD_ACCUM=1: the pinned DeepSpeed 0.17.6 has a known ZeRO-2 accumulation bug (#7718).' >&2
    exit 1
fi
MAX_STEPS="${MAX_STEPS:-60000}"
RUN_NAME="${RUN_NAME:-rldx1_img_robocasa_gb64_60k_baseline}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_OUTPUT_DIR:-$BASE_DIR/outputs}}"
[[ -f "$DATA_DIR/meta/modality.json" ]] || { echo "Missing dataset: $DATA_DIR" >&2; exit 1; }
[[ "$GRAD_ACCUM" =~ ^[1-9][0-9]*$ ]] && (( GLOBAL_BATCH_SIZE % (NUM_GPUS * GRAD_ACCUM) == 0 )) || {
    echo 'GRAD_ACCUM must be a positive divisor of 32.' >&2; exit 1;
}
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS=1
# NCCL 2.26.2 lacks the host-cuMem allocation fallback added in 2.26.5.
# Apply NVIDIA's documented /dev/shm fallback only on the H100 partition.
# Keep this in the shared runtime script so queued H100 jobs pick it up.
if [[ "${SLURM_JOB_PARTITION:-}" == h100 ]]; then
    export NCCL_CUMEM_HOST_ENABLE=0
    echo 'H100 NCCL: host cuMem disabled; using /dev/shm allocation'
fi
export RLDX_COMPILE_RMSNORM="${RLDX_COMPILE_RMSNORM:-1}"
export RLDX_CPU_ROPE="${RLDX_CPU_ROPE:-1}"
export RLDX_IMAGE_IDENTITY_FASTPATH="${RLDX_IMAGE_IDENTITY_FASTPATH:-1}"
export RLDX_REFRESH_SAVE_INTERVAL=1
export RLDX_LOG_THROUGHPUT=1
export NCCL_PROTO="${NCCL_PROTO:-Simple}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$BASE_DIR/.cache/uv}"
export HF_HOME="${HF_HOME:-$BASE_DIR/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$BASE_DIR/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$BASE_DIR/.cache/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$BASE_DIR/.cache/torch_extensions}"
# Invoke the installed environment directly: training must not resolve/install packages.
export PATH="$BASE_DIR/.venv/bin:$PATH"
echo "Data loader: sharded, workers per rank=${NUM_WORKERS:-12}, decoder threads=1"
echo "Training kernels: compiled RMSNorm=$RLDX_COMPILE_RMSNORM CPU RoPE=$RLDX_CPU_ROPE image identity=$RLDX_IMAGE_IDENTITY_FASTPATH NCCL=$NCCL_PROTO save interval=${SAVE_STEPS:-2000}"
echo "Image checkpoint=$BASE_MODEL_PATH frames=1 global_batch=$GLOBAL_BATCH_SIZE GPUs=$NUM_GPUS accumulation=$GRAD_ACCUM microbatch=$((GLOBAL_BATCH_SIZE / NUM_GPUS / GRAD_ACCUM))"
exec "$BASE_DIR/.venv/bin/torchrun" --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
    "${TRAIN_ENTRYPOINT:-rldx/experiment/launch_train.py}" \
    --base-model-path "$BASE_MODEL_PATH" --video-length 1 --n-cog-tokens 64 --action-horizon 16 \
    --dataset-path "$DATA_DIR" --dataset-mode sharded \
    --dataloader-num-workers "${NUM_WORKERS:-12}" \
    --embodiment-tag GENERAL_EMBODIMENT \
    --modality-config-path "$BASE_DIR/rldx/configs/data/robocasa_config.py" \
    --rtc-training-max-delay 0 --rtc-inference-mode none \
    --state-dropout-prob 0.0 --num-gpus "$NUM_GPUS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" --gradient-accumulation-steps "$GRAD_ACCUM" \
    --max-steps "$MAX_STEPS" --save-steps "${SAVE_STEPS:-2000}" --save-total-limit 2 \
    --output-dir "$OUTPUT_DIR" --experiment-name "$RUN_NAME"
