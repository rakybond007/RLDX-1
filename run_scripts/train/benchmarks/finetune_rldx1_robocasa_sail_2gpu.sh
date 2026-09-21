#!/bin/bash
#SBATCH --job-name=train_rldx1_pt_image_robocasa_sail_eag_precision_60k_globalbatch64_2gpu
#SBATCH --comment="RLDX-1 image + SAIL EAG (future-action CFG) + precision bit, current-frame input, action chunk 16, no gradient accumulation"
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --time=2-00:00:00
#SBATCH --output=out/%j-rldx1_robocasa_sail.out
#SBATCH --error=out/%j-rldx1_robocasa_sail.err

# SAIL reproduction on RLDX-1-PT-IMG: base + EAG + precision bit.
# Differs from finetune_rldx1_robocasa_image_2gpu.sh in exactly four places:
#   1. dataset            -> the SAIL delta dataset carrying action.precision
#   2. modality config    -> robocasa_sail_config.py (baseline + precision column)
#   3. EAG flags          -> the four --*future-action-condition* / --eag-* knobs
#   4. --save-total-limit -> 5, so checkpoint-20000 survives until step 25000
#                            and the keep-ckpt watcher jobs can copy it out.
#
# Smoke (2-GPU srun shell):  MAX_STEPS=10 SAVE_STEPS=10 bash <this script>
# Full (slurm):              mkdir -p out; sbatch <this script>
# Full (MLXP):               the manifest sets DATA_DIR/OUTPUT_DIR and bashes this file.
set -euo pipefail
BASE_DIR="${RLDX_ROOT:-$(pwd)}"
if [[ ! -f "$BASE_DIR/rldx/experiment/launch_train.py" ]]; then
    echo 'Run from the RLDX-1 repository root, or set RLDX_ROOT.' >&2
    exit 1
fi
cd "$BASE_DIR"
DATA_DIR="${DATA_DIR:-/sjw_alinlab2/home/taekwan/sail_robocasa_preprocess/sail_robocasa_delta_lerobot}"
if [[ -z "${BASE_MODEL_PATH:-}" ]]; then
    if [[ -f "$BASE_DIR/models/RLDX-1-PT-IMG/model.safetensors.index.json" ]]; then
        BASE_MODEL_PATH="$BASE_DIR/models/RLDX-1-PT-IMG"
    else
        BASE_MODEL_PATH="RLWRLD/RLDX-1-PT-IMG"
    fi
fi
NUM_GPUS="${NUM_GPUS:-2}"
GLOBAL_BATCH_SIZE=64
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-60000}"
RUN_NAME="${RUN_NAME:-rldx1_img_robocasa_sail_eag_gb64_60k}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_OUTPUT_DIR:-$BASE_DIR/outputs}}"
[[ -f "$DATA_DIR/meta/modality.json" ]] || { echo "Missing dataset: $DATA_DIR" >&2; exit 1; }
[[ "$GRAD_ACCUM" =~ ^[1-9][0-9]*$ ]] && (( GLOBAL_BATCH_SIZE % (NUM_GPUS * GRAD_ACCUM) == 0 )) || {
    echo "GRAD_ACCUM must be a positive divisor of $((GLOBAL_BATCH_SIZE / NUM_GPUS))." >&2; exit 1;
}
# The precision column must actually be in the dataset, or we would train the
# baseline under a SAIL run name and only find out at eval.
python3 - "$DATA_DIR/meta/modality.json" <<'PYEOF'
import json, sys
action = json.load(open(sys.argv[1]))["action"]
if "precision" not in action:
    sys.exit(f"[FATAL] {sys.argv[1]} has no action.precision; this is not a SAIL dataset")
print(f"[i] action.precision at columns {action['precision']['start']}:{action['precision']['end']}")
PYEOF
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$BASE_DIR/.cache/uv}"
export HF_HOME="${HF_HOME:-$BASE_DIR/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$BASE_DIR/.cache/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$BASE_DIR/.cache/torch_extensions}"
# Invoke the installed environment directly: training must not resolve/install packages.
export PATH="$BASE_DIR/.venv/bin:$PATH"
echo "Image checkpoint=$BASE_MODEL_PATH frames=1 global_batch=$GLOBAL_BATCH_SIZE GPUs=$NUM_GPUS accumulation=$GRAD_ACCUM microbatch=$((GLOBAL_BATCH_SIZE / NUM_GPUS / GRAD_ACCUM))"
exec "$BASE_DIR/.venv/bin/torchrun" --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
    rldx/experiment/launch_train.py \
    --base-model-path "$BASE_MODEL_PATH" --video-length 1 --n-cog-tokens 64 --action-horizon 16 \
    --dataset-path "$DATA_DIR" --dataset-mode standard \
    --dataloader-num-workers "${NUM_WORKERS:-4}" \
    --embodiment-tag GENERAL_EMBODIMENT \
    --modality-config-path "$BASE_DIR/rldx/configs/data/robocasa_sail_config.py" \
    --use-future-action-condition \
    --future-action-condition-horizon 4 \
    --future-action-condition-dropout 0.1 \
    --eag-cfg-weight 1.0 \
    --state-dropout-prob 0.0 --num-gpus "$NUM_GPUS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" --gradient-accumulation-steps "$GRAD_ACCUM" \
    --max-steps "$MAX_STEPS" --save-steps "${SAVE_STEPS:-1000}" --save-total-limit 5 \
    --output-dir "$OUTPUT_DIR" --experiment-name "$RUN_NAME"
