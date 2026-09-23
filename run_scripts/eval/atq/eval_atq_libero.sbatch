#!/bin/bash
#SBATCH --job-name=eval_rldx1_atq_libero_labelgated_speed167_confboth_40tasks_all_suites_jimin
#SBATCH --partition=sjw_alinlab_premium,h100
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --output=out/%j-eval_atq_libero.out
#SBATCH --error=out/%j-eval_atq_libero.out
# LIBERO eval for an ATQ label-gated checkpoint. See eval_atq_robocasa.sbatch
# for the two ATQ constraints; N_ENVS must stay 1 here (the stock script uses 5).
set -u
cd /sjw_alinlab2/home/jimin/workspace/FasterVLA/RLDX-1-ATQ
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NO_ALBUMENTATIONS_UPDATE=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}" HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export PATH="$PWD/.venv/bin:$PATH"
RUN_LABEL="${RUN_LABEL:?set RUN_LABEL}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
[[ -d "$MODEL_PATH" ]] || { echo "[ERR] missing checkpoint: $MODEL_PATH" >&2; exit 1; }
export N_ENVS="${N_ENVS:-1}" N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1500}"
echo "=== ATQ LIBERO eval | label=$RUN_LABEL | ckpt=$MODEL_PATH"
echo "=== n_envs=$N_ENVS n_action_steps=$N_ACTION_STEPS max_episode_steps=$MAX_EPISODE_STEPS"
bash run_scripts/eval/libero/eval_libero.sh "$RUN_LABEL" "$MODEL_PATH" 0 "${MAX_PARALLEL:-4}"
RC=$?
echo "=== eval rc=$RC $(date '+%F %T')"
python3 run_scripts/eval/atq/summarize_eval.py --root "output_final/libero/$RUN_LABEL" --benchmark libero || true
exit $RC
