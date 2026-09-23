#!/bin/bash
#SBATCH --job-name=train_rldx1_pt_image_robocasa_atq_labelgated_60k_gb64_2gpu
#SBATCH --comment="RLDX-1 image ATQ label-gated MoE, current-frame input, action chunk 16, 4 experts"
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab_premium
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --time=2-00:00:00
#SBATCH --output=out/%j-rldx1_img_robocasa_atq.out
#SBATCH --error=out/%j-rldx1_img_robocasa_atq.err

# ATQ label-gated variable-horizon MoE on top of the RoboCasa image baseline
# (run_scripts/train/benchmarks/finetune_rldx1_robocasa_image_2gpu.sh).
# Same base checkpoint, batch, schedule and data; adds the 4-expert MSAT head.
#
#   SPEED=2.0|2.5        compressed-group speed (block plan)
#   VARIANT=vlm|vlm_contact|contact   which baked conf dataset to read
#   ROTATION_MERGE=legacy|so3         summed rotation deltas (upstream gpu26 run) vs SO(3) composition
#   INIT_EXPERTS_FROM_MAIN=0|1        0 = random-init m8/m4/n8 decoders (upstream gpu26 run), 1 = copy main
#   TAU=0.5              conf threshold written into the checkpoint (eval knob)
#
# The label is baked into the dataset (run_scripts/data/atq_labels/README.md),
# so VARIANT only picks the path: $CONF_DATA_ROOT/robocasa_conf_$VARIANT.
#
# From repo root: mkdir -p out; SPEED=2.0 VARIANT=vlm_contact sbatch <this script>
# Smoke in an existing 2-GPU srun shell:
#   SPEED=2.0 VARIANT=vlm_contact MAX_STEPS=10 SAVE_STEPS=10 bash <this script>
set -euo pipefail
BASE_DIR="${RLDX_ROOT:-$(pwd)}"
if [[ ! -f "$BASE_DIR/rldx/experiment/launch_train.py" ]]; then
    echo 'Run from the RLDX-1 repository root, or set RLDX_ROOT.' >&2
    exit 1
fi
cd "$BASE_DIR"

SPEED="${SPEED:?set SPEED=2.0 or 2.5}"
VARIANT="${VARIANT:?set VARIANT=vlm|vlm_contact|contact}"
ROTATION_MERGE="${ROTATION_MERGE:-legacy}"
INIT_EXPERTS_FROM_MAIN="${INIT_EXPERTS_FROM_MAIN:-0}"
TAU="${TAU:-0.5}"
CONF_DATA_ROOT="${CONF_DATA_ROOT:-/sjw_alinlab2/home/jimin/workspace/dataset}"
DATA_DIR="${DATA_DIR:-$CONF_DATA_ROOT/robocasa_conf_$VARIANT}"

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
MAX_STEPS="${MAX_STEPS:-60000}"
SPEED_TAG="${SPEED//./}"
INIT_TAG=$([[ "$INIT_EXPERTS_FROM_MAIN" == "1" ]] && echo initmain || echo initrand)
INIT_FLAG=$([[ "$INIT_EXPERTS_FROM_MAIN" == "1" ]] && echo --atq-init-experts-from-main || echo --no-atq-init-experts-from-main)
RUN_NAME="${RUN_NAME:-rldx1_img_robocasa_atq_labelgated_${SPEED_TAG}_${VARIANT}_${ROTATION_MERGE}_${INIT_TAG}_gb64_60k}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_OUTPUT_DIR:-$BASE_DIR/outputs}}"

[[ -f "$DATA_DIR/meta/modality.json" ]] || { echo "Missing dataset: $DATA_DIR" >&2; exit 1; }
# The baked dataset must carry the conf carrier and say which variant it is.
python3 - "$DATA_DIR" "$VARIANT" <<'EOF'
import json, sys
root, variant = sys.argv[1], sys.argv[2]
mod = json.load(open(f"{root}/meta/modality.json"))
assert "ratio_label" in mod["action"], f"{root}: modality.json has no action.ratio_label (not baked?)"
info = json.load(open(f"{root}/meta/info.json"))
got = info.get("conf_variant")
assert got == variant, f"{root}: dataset says conf_variant={got!r}, asked {variant!r}"
print(f"[data] ratio_label at [{mod['action']['ratio_label']['start']},{mod['action']['ratio_label']['end']}), conf_variant={got}")
EOF
[[ "$GRAD_ACCUM" =~ ^[1-9][0-9]*$ ]] && (( GLOBAL_BATCH_SIZE % (NUM_GPUS * GRAD_ACCUM) == 0 )) || {
    echo 'GRAD_ACCUM must be a positive divisor of 32.' >&2; exit 1;
}
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
# NCCL 2.26.2 lacks the host-cuMem allocation fallback added in 2.26.5.
# Apply NVIDIA's documented /dev/shm fallback only on the H100 partition
# (without it every 2-GPU job on an H100 node dies at the first NCCL barrier
# with "CUDA error: out of memory"). Mirrors hojin2's baseline runtime script.
if [[ "${SLURM_JOB_PARTITION:-}" == h100 ]]; then
    export NCCL_CUMEM_HOST_ENABLE=0
    echo 'H100 NCCL: host cuMem disabled; using /dev/shm allocation'
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$BASE_DIR/.cache/uv}"
export HF_HOME="${HF_HOME:-$BASE_DIR/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$BASE_DIR/.cache/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$BASE_DIR/.cache/torch_extensions}"
export PATH="$BASE_DIR/.venv/bin:$PATH"
echo "Data loader: sharded, workers per rank=${NUM_WORKERS:-12}, decoder threads=1"
echo "ATQ checkpoint=$BASE_MODEL_PATH data=$DATA_DIR speed=$SPEED rotation=$ROTATION_MERGE init_experts_from_main=$INIT_EXPERTS_FROM_MAIN tau=$TAU global_batch=$GLOBAL_BATCH_SIZE GPUs=$NUM_GPUS accumulation=$GRAD_ACCUM microbatch=$((GLOBAL_BATCH_SIZE / NUM_GPUS / GRAD_ACCUM))"
exec "$BASE_DIR/.venv/bin/torchrun" --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
    rldx/experiment/launch_train.py \
    --base-model-path "$BASE_MODEL_PATH" --video-length 1 --n-cog-tokens 64 --action-horizon 16 \
    --dataset-path "$DATA_DIR" --dataset-mode sharded \
    --dataloader-num-workers "${NUM_WORKERS:-12}" \
    --embodiment-tag GENERAL_EMBODIMENT \
    --modality-config-path "$BASE_DIR/rldx/configs/data/robocasa_conf_config.py" \
    --state-dropout-prob 0.0 --num-gpus "$NUM_GPUS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" --gradient-accumulation-steps "$GRAD_ACCUM" \
    --max-steps "$MAX_STEPS" --save-steps "${SAVE_STEPS:-1000}" --save-total-limit 2 \
    --use-atq-moe --atq-speed "$SPEED" --atq-discrete-action-dims 6 11 \
    --atq-rotation-merge "$ROTATION_MERGE" --atq-rotation-key end_effector_rotation \
    --atq-rotation-controller-scale 0.5 \
    --atq-label-gated --atq-conf-carrier-key ratio_label --atq-conf-threshold "$TAU" \
    --atq-conf-loss-coef 0.1 "$INIT_FLAG" \
    --output-dir "$OUTPUT_DIR" --experiment-name "$RUN_NAME"
