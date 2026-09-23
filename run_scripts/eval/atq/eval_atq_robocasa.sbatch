#!/bin/bash
#SBATCH --job-name=eval_rldx1_atq_robocasa300_labelgated_speed20_vlmcontact_24tasks_50ep_jimin
#SBATCH --partition=sjw_alinlab_premium,h100
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=1-00:00:00
#SBATCH --output=out/%j-eval_atq_robocasa.out
#SBATCH --error=out/%j-eval_atq_robocasa.out
# RoboCasa Kitchen eval for an ATQ label-gated checkpoint.
#   RUN_LABEL  output tree name under output_final/robocasa/
#   MODEL_PATH trained checkpoint directory
# ATQ constraints: one env per request (PolicyRuntime._decode_atq rejects mixed
# expert picks in a batch), and the rollout executes the rows the picked expert
# returned (MultiStepWrapper caps at N_ACTION_STEPS).
set -u
cd /sjw_alinlab2/home/jimin/workspace/FasterVLA/RLDX-1-ATQ
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NO_ALBUMENTATIONS_UPDATE=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}" HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export PATH="$PWD/.venv/bin:$PATH"
RUN_LABEL="${RUN_LABEL:?set RUN_LABEL}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
[[ -d "$MODEL_PATH" ]] || { echo "[ERR] missing checkpoint: $MODEL_PATH" >&2; exit 1; }
export N_ENVS="${N_ENVS:-1}" N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
export N_EPISODES="${N_EPISODES:-50}" MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1500}"
echo "=== ATQ RoboCasa eval | label=$RUN_LABEL | ckpt=$MODEL_PATH"
echo "=== n_envs=$N_ENVS n_action_steps=$N_ACTION_STEPS n_episodes=$N_EPISODES max_episode_steps=$MAX_EPISODE_STEPS"
bash run_scripts/eval/robocasa_kitchen/eval_robocasa.sh "$RUN_LABEL" "$MODEL_PATH"
RC=$?
echo "=== eval rc=$RC $(date '+%F %T')"
python3 run_scripts/eval/atq/summarize_eval.py --root "output_final/robocasa/$RUN_LABEL" --benchmark robocasa || true
exit $RC
