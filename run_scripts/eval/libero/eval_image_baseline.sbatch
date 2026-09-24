#!/bin/bash
#SBATCH --job-name=eval_rldx1_image_libero_checkpoint60000_replan5_50episodes
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=sjw_alinlab_premium
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --array=0-3%2
#SBATCH --time=1-00:00:00
#SBATCH --output=out/%A_%a-rldx1_libero_eval.out
#SBATCH --error=out/%A_%a-rldx1_libero_eval.err
set -euo pipefail
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME="$PWD/.cache/huggingface"
export LIBERO_CONFIG_PATH="$PWD/.cache/libero-eval"
export PYTHONPATH="$PWD:$PWD/external_dependencies/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
CKPT="${CKPT:-/rlwrld-unified-checkpoints/hojin2/checkpoints/rldx1_image_baselines/rldx1_img_libero_gb32_60k_baseline/checkpoint-60000}"
SUITES=(libero_spatial libero_object libero_goal libero_10)
SUITE="${SUITES[${SLURM_ARRAY_TASK_ID:-0}]}"
RESULT_TAG="${RESULT_TAG:-image_60000_replan5}"
RESULTS="$PWD/output_final/libero/${RESULT_TAG}_${SLURM_ARRAY_JOB_ID:-local}/$SUITE"
mkdir -p "$RESULTS"
PORT=$(.venv/bin/python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
.venv/bin/python rldx/eval/run_rldx_server.py --model-path "$CKPT" \
    --embodiment-tag GENERAL_EMBODIMENT --use-sim-policy-wrapper --no-strict \
    --rtc-inference-mode none --host 127.0.0.1 --port "$PORT" > "$RESULTS/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
EVAL_ARGS=()
if [[ "${NO_ACTION_CLIP:-0}" == 1 ]]; then EVAL_ARGS+=(--no-action-clip); fi
EVAL_ARGS+=(--gripper-gain-scale "${GRIPPER_GAIN_SCALE:-1.0}")
.venv-libero-eval/bin/python run_scripts/eval/libero/eval_image_baseline.py \
    --suite "$SUITE" --port "$PORT" --output "$RESULTS" --episodes 50 --replan-steps 5 --seed 7 "${EVAL_ARGS[@]}"
