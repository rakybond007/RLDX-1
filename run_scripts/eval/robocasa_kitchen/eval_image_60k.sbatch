#!/bin/bash
#SBATCH --job-name=eval_rldx1_robocasa_image_gb64_checkpoint60000_kitchen_24tasks
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=sjw_alinlab_premium
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --array=0-3%2
#SBATCH --time=2-00:00:00
#SBATCH --output=out/%A_%a-rldx1_robocasa_image_eval.out
#SBATCH --error=out/%A_%a-rldx1_robocasa_image_eval.err
set -euo pipefail
export PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME="$PWD/.cache/huggingface"
export PYTHONPATH="$PWD:$PWD/.work/robocasa_local:$PWD/.work/robosuite_local${PYTHONPATH:+:$PYTHONPATH}"
CKPT="$PWD/models/rldx1-robocasa-img-gb64-60k-baseline"
test -f "$CKPT/model.safetensors.index.json"
test -x .venv-robocasa-eval/bin/python
test -d .work/robocasa_local/robocasa/models/assets/objects/objaverse
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
TASKS=(
  TurnSinkSpout TurnOnStove TurnOnSinkFaucet TurnOnMicrowave TurnOffStove TurnOffSinkFaucet
  TurnOffMicrowave PnPStoveToCounter PnPSinkToCounter PnPMicrowaveToCounter PnPCounterToStove PnPCounterToSink
  PnPCounterToMicrowave PnPCounterToCab PnPCabToCounter OpenSingleDoor OpenDrawer OpenDoubleDoor
  CoffeeSetupMug CoffeeServeMug CoffeePressButton CloseSingleDoor CloseDrawer CloseDoubleDoor
)
RESULTS="$PWD/output_final/robocasa/image_60000_ac16_${SLURM_ARRAY_JOB_ID:-local}"
mkdir -p "$RESULTS"
PORT=$(.venv/bin/python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
.venv/bin/python rldx/eval/run_rldx_server.py --model-path "$CKPT" \
    --embodiment-tag GENERAL_EMBODIMENT --use-sim-policy-wrapper --no-strict \
    --rtc-inference-mode none --host 127.0.0.1 --port "$PORT" \
    > "$RESULTS/server_${SHARD}.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

# Wait for model load and socket bind; fail if the server exits.
ready=0
for attempt in $(seq 1 240); do
    if .venv/bin/python -c 'import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(s.connect_ex(("127.0.0.1",int(sys.argv[1]))))' "$PORT"; then
        ready=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Model server exited before binding the port" >&2
        exit 1
    fi
    sleep 5
done
if [[ "$ready" != 1 ]] || ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Model server did not become ready" >&2
    exit 1
fi

for offset in 0 1 2 3 4 5; do
    index=$((SHARD * 6 + offset))
    task="${TASKS[$index]}"
    target="$RESULTS/$task"
    mkdir -p "$target"
    echo "ROBOCASA_TASK_START shard=$SHARD task=$task"
    RLDX_SKIP_HF_REGISTRATION=1 .venv-robocasa-eval/bin/python \
        rldx/eval/rollout_policy.py --n_episodes 50 \
        --policy_client_host 127.0.0.1 --policy_client_port "$PORT" \
        --max_episode_steps 720 \
        --env_name "robocasa_panda_omron/${task}_PandaOmron_Env" \
        --n_action_steps 16 --n_envs 1 --seed 42 --video_dir "$target" \
        > "$target/eval.log" 2>&1
    test "$(tail -n 1 "$target/eval.log" | cut -c 1-13)" = 'success rate:' || {
        echo "No success-rate line for $task" >&2
        exit 1
    }
    echo "ROBOCASA_TASK_COMPLETE shard=$SHARD task=$task"
done
echo "ROBOCASA_SHARD_COMPLETE shard=$SHARD"
