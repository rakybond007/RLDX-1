#!/bin/bash
#SBATCH --job-name=train_rldx1_pt_image_libero_atq_labelgated_60k_globalbatch32_2gpu
#SBATCH --comment="RLDX-1 image ATQ label-gated MoE on LIBERO, current-frame input, action chunk 16, 4 experts"
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --partition=sjw_alinlab_premium
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --time=2-00:00:00
#SBATCH --output=out/%j-rldx1_img_libero_atq.out
#SBATCH --error=out/%j-rldx1_img_libero_atq.err

# ATQ label-gated variable-horizon MoE on top of the LIBERO image baseline
# (finetune_rldx1_libero_image_2gpu.sh). Same base checkpoint, batch, schedule
# and data; adds the 4-expert MSAT head. Mirrors GR00T-action-quantization's
# run_scripts/gpu26/atq_libero_train.sh (v3c labels, gate conf_both, tau 0.40).
#
#   SPEED=1.67|2.5       compressed-group speed (1.67 = LIBERO's 5-step replan rhythm [2,2,1])
#   GATE=both|vlm|contact  which baked conf_<gate> dataset to read
#   TAU=0.40             conf threshold written into the checkpoint (eval knob)
#   DISCRETE_DIMS="6"    last-of-block dims (gripper_close). "" = none (upstream LIBERO run summed it)
#   ROTATION_MERGE=legacy  eef_rot_delta is euler rpy, so SO(3) composition is not applicable
#   INIT_EXPERTS_FROM_MAIN=0|1
#
# From repo root: mkdir -p out; SPEED=1.67 GATE=both sbatch <this script>
# Smoke in an existing 2-GPU srun shell: SPEED=1.67 GATE=both MAX_STEPS=10 SAVE_STEPS=10 bash <this script>
set -euo pipefail
BASE_DIR="${RLDX_ROOT:-$(pwd)}"
if [[ ! -f "$BASE_DIR/rldx/experiment/launch_train.py" ]]; then
    echo 'Run from the RLDX-1 repository root, or set RLDX_ROOT.' >&2
    exit 1
fi
cd "$BASE_DIR"
SPEED="${SPEED:?set SPEED=1.67 or 2.5}"
GATE="${GATE:-both}"
TAU="${TAU:-0.40}"
DISCRETE_DIMS="${DISCRETE_DIMS-6}"
ROTATION_MERGE="${ROTATION_MERGE:-legacy}"
INIT_EXPERTS_FROM_MAIN="${INIT_EXPERTS_FROM_MAIN:-0}"
CONF_DATA_ROOT="${CONF_DATA_ROOT:-/sjw_alinlab2/home/jimin/workspace/dataset}"
DATA_DIR="${DATA_DIR:-$CONF_DATA_ROOT/libero_confv3c_$GATE}"
if [[ -z "${BASE_MODEL_PATH:-}" ]]; then
    if [[ -f "$BASE_DIR/models/RLDX-1-PT-IMG/model.safetensors.index.json" ]]; then
        BASE_MODEL_PATH="$BASE_DIR/models/RLDX-1-PT-IMG"
    else
        BASE_MODEL_PATH="RLWRLD/RLDX-1-PT-IMG"
    fi
fi
NUM_GPUS=2
GLOBAL_BATCH_SIZE=32
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-60000}"
SPEED_TAG="${SPEED//./}"
INIT_TAG=$([[ "$INIT_EXPERTS_FROM_MAIN" == "1" ]] && echo initmain || echo initrand)
INIT_FLAG=$([[ "$INIT_EXPERTS_FROM_MAIN" == "1" ]] && echo --atq-init-experts-from-main || echo --no-atq-init-experts-from-main)
DISC_TAG=$([[ -n "$DISCRETE_DIMS" ]] && echo "disc${DISCRETE_DIMS// /}" || echo nodisc)
RUN_NAME="${RUN_NAME:-rldx1_img_libero_atq_labelgated_${SPEED_TAG}_conf${GATE}_${ROTATION_MERGE}_${INIT_TAG}_${DISC_TAG}_gb32_60k}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_OUTPUT_DIR:-$BASE_DIR/outputs}}"
[[ -f "$DATA_DIR/meta/modality.json" ]] || { echo "Missing dataset: $DATA_DIR" >&2; exit 1; }
# The baked dataset must carry the conf carrier and say which gate it is.
python3 - "$DATA_DIR" "conf_$GATE" <<'PYEOF'
import json, sys
root, gate = sys.argv[1], sys.argv[2]
mod = json.load(open(f"{root}/meta/modality.json"))
assert "ratio_label" in mod["action"], f"{root}: modality.json has no action.ratio_label (not baked?)"
info = json.load(open(f"{root}/meta/info.json"))
got = info.get("conf_variant")
assert got == gate, f"{root}: dataset says conf_variant={got!r}, asked {gate!r}"
print(f"[data] ratio_label at [{mod['action']['ratio_label']['start']},{mod['action']['ratio_label']['end']}), conf_variant={got}, release={info.get('label_release')}")
PYEOF
[[ "$GRAD_ACCUM" =~ ^[1-9][0-9]*$ ]] && (( GLOBAL_BATCH_SIZE % (NUM_GPUS * GRAD_ACCUM) == 0 )) || {
    echo 'GRAD_ACCUM must be a positive divisor of 16.' >&2; exit 1;
}
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$BASE_DIR/.cache/uv}"
export HF_HOME="${HF_HOME:-$BASE_DIR/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$BASE_DIR/.cache/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$BASE_DIR/.cache/torch_extensions}"
# Invoke the installed environment directly: training must not resolve/install packages.
export PATH="$BASE_DIR/.venv/bin:$PATH"
echo "Data loader: sharded, workers per rank=${NUM_WORKERS:-8}, decoder threads=1"
echo "ATQ speed=$SPEED gate=conf_$GATE tau=$TAU discrete_dims=[$DISCRETE_DIMS] rotation=$ROTATION_MERGE init_experts_from_main=$INIT_EXPERTS_FROM_MAIN"
echo "Image checkpoint=$BASE_MODEL_PATH frames=1 global_batch=$GLOBAL_BATCH_SIZE GPUs=$NUM_GPUS accumulation=$GRAD_ACCUM microbatch=$((GLOBAL_BATCH_SIZE / NUM_GPUS / GRAD_ACCUM))"
exec "$BASE_DIR/.venv/bin/torchrun" --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
    "${TRAIN_ENTRYPOINT:-rldx/experiment/launch_train.py}" \
    --base-model-path "$BASE_MODEL_PATH" --video-length 1 --n-cog-tokens 64 --action-horizon 16 \
    --dataset-path "$DATA_DIR" --dataset-mode sharded \
    --dataloader-num-workers "${NUM_WORKERS:-8}" \
    --embodiment-tag GENERAL_EMBODIMENT \
    --modality-config-path "$BASE_DIR/rldx/configs/data/libero_conf_config.py" \
    --rtc-training-max-delay 0 --rtc-inference-mode none \
    --state-dropout-prob 0.0 --num-gpus "$NUM_GPUS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" --gradient-accumulation-steps "$GRAD_ACCUM" \
    --max-steps "$MAX_STEPS" --save-steps "${SAVE_STEPS:-1000}" --save-total-limit 2 \
    --use-atq-moe --atq-speed "$SPEED" ${DISCRETE_DIMS:+--atq-discrete-action-dims $DISCRETE_DIMS} \
    --atq-rotation-merge "$ROTATION_MERGE" --atq-rotation-key eef_rot_delta \
    --atq-label-gated --atq-conf-carrier-key ratio_label --atq-conf-threshold "$TAU" \
    --atq-conf-loss-coef 0.1 "$INIT_FLAG" \
    --output-dir "$OUTPUT_DIR" --experiment-name "$RUN_NAME"
