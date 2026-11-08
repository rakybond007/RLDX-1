#!/bin/bash
#SBATCH --job-name=eval_rldx1_img_sail_eag_precision_robocasa_24task_through_the_rldx_rollout_client_final_kp
#SBATCH --comment="RoboCasa 24-task eval of the RLDX-1-img SAIL checkpoint (EAG + precision bit, step 60000) through the RLDX rollout client scene (object split B, original textures, five layout-style pairs), SAIL loop ported into rldx/eval/sail_rollout_robocasa.py: replan 12, EAG horizon 4 with the goal gate, precision window 6, c_slow 1.0, c_fast 0.5, per-task final kp, paper w 1 and w 0 arms, fifty episodes per task, seed forty two, per-episode resume."
#SBATCH --partition=background
#SBATCH --gres=gpu:1
#SBATCH --exclude=worker-node100,worker-node1001,worker-node104
#SBATCH --wckey=project-short-name:sub_fast
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-7%8
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -uo pipefail

# Same conditions as Isaac-GR00T run_scripts/eval/eval_robocasa_rldx_sail_finalkp_w01_24task.sh
# (job 206368/206377); only the client differs: rldx/eval/sail_rollout_robocasa.py builds the env
# with rollout_policy.create_eval_env (RoboCasa's own eval scene).
#
# One cell = one GPU running BOTH arms: two policy servers (paper w 1 = code EAG_CFG_W 0.0,
# paper w 0 = code -1.0) and 3 tasks per arm (6 clients). 8 cells cover 24 tasks x 2 arms.
# Submit:
#   MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/taekwan/sail_robocasa_rldx_rldxclient \
#     sbatch --wckey=project-short-name:sub_fast run_scripts/eval/sail/eval_sail_robocasa_rldxclient.sbatch
# Resume: re-submit; every task continues after the last line of its prediction.txt.

# ════════════════════════ 고치는 곳 ════════════════════════
SEED=${SEED:-42}
N_EP=${N_EP:-50}
MAX_EP_STEPS=${MAX_EP_STEPS:-1500}
REPLAN=${REPLAN:-12}
EAG_HORIZON=${EAG_HORIZON:-4}
EAG_GATE=${EAG_GATE:-goal}
POS_TEB=${POS_TEB:-0.02}
ORI_TEB=${ORI_TEB:-0.05}
C_SLOW=${C_SLOW:-1.00}
C_FAST=${C_FAST:-0.50}
PRECISION_THR=${PRECISION_THR:-0.5}
SLOWDOWN_WINDOW=${SLOWDOWN_WINDOW:-6}
DAMPING=${DAMPING:-1.0}
MINUTES=${MINUTES:-225}
PORT_BASE=${PORT_BASE:-21700}
TASK_LIST=${TASK_LIST:-}          # space-separated task names (smoke); needs ARM_IDX
ARM_W=("0.0" "-1.0")
ARM_NAME=("paperw1" "paperw0")

CKPT=${CKPT:-/sjw_alinlab2/home/taekwan/ckpt/rldx_img_sail/robocasa/step60000}
RLDX_ROOT=/sjw_alinlab2/home/taekwan/reproducing/RLDX-1                             # tk branch: EAG server + this client
SERVER_PY=$RLDX_ROOT/.venv/bin/python
SERVER_HF_HOME=$RLDX_ROOT/.cache/huggingface
CLIENT_PY=/sjw_alinlab2/home/jimin/workspace/FasterVLA/RLDX-1-ATQ/rldx/eval/sim/robocasa/robocasa_uv/.venv/bin/python

# ═════════════════════ 아래는 실행 코드 ═════════════════════
# task:kp — final kp (GR00T-action-quantization/analysis/final_kp/FINAL_KP.md). 150 = env default.
TASKCFG=(
  "OpenDoubleDoor:150" "CloseDoubleDoor:150" "CloseDrawer:300" "CoffeeServeMug:150"
  "PnPCounterToCab:150" "TurnOffSinkFaucet:150" "TurnOffStove:150" "CloseSingleDoor:300"
  "TurnSinkSpout:600" "CoffeePressButton:150" "OpenSingleDoor:150" "PnPCounterToSink:300"
  "PnPSinkToCounter:300" "CoffeeSetupMug:150" "OpenDrawer:150" "PnPCounterToStove:300"
  "PnPCabToCounter:300" "TurnOnStove:150" "TurnOnMicrowave:150" "PnPMicrowaveToCounter:300"
  "TurnOnSinkFaucet:150" "TurnOffMicrowave:300" "PnPStoveToCounter:300" "PnPCounterToMicrowave:150"
)
CHUNKS=("0 1 2" "3 4 5" "6 7 8" "9 10 11" "12 13 14" "15 16 17" "18 19 20" "21 22 23")

BASE_DIR=/sjw_alinlab2/home/taekwan/reproducing
source "$BASE_DIR/run_scripts/eval/_model_output_dir.sh" || exit 1
source "$BASE_DIR/run_scripts/eval/_wait_server.sh" || exit 1
eqs_require_model_output_dir

IDX=${SLURM_ARRAY_TASK_ID:-0}
[ "$IDX" -lt "${#CHUNKS[@]}" ] || { echo "[i] cell $IDX: no chunk"; exit 0; }
ARMS=(0 1); [ -n "${ARM_IDX:-}" ] && ARMS=("$ARM_IDX")
T0=$(date +%s); DEADLINE=$((T0 + MINUTES * 60))
TAG0="rldxclient_sail_finalkp50_cs100_cf050_z10"
LOG=$(eqs_logfile "rldxclient_sail_cell${IDX}.log")
exec >>"$LOG" 2>&1

SEL=()
if [ -n "$TASK_LIST" ]; then
  for name in $TASK_LIST; do
    for i in "${!TASKCFG[@]}"; do [ "${TASKCFG[$i]%%:*}" = "$name" ] && SEL+=("$i"); done
  done
else
  read -r -a SEL <<< "${CHUNKS[$IDX]}"
fi

echo "[$(date '+%F %T')] cell=$IDX node=$(hostname) arms=${ARMS[*]} tasks=${SEL[*]}"
echo "  ckpt=$CKPT"
[ -d "$CKPT" ] || { echo "[ERR] no checkpoint: $CKPT"; exit 1; }

export NO_ALBUMENTATIONS_UPDATE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HOME=/sjw_alinlab/home/hojin2
for PY in "$SERVER_PY" "$CLIENT_PY"; do
  PYTHONPATH="$RLDX_ROOT" "$PY" -c "import rldx,sys; assert rldx.__file__.startswith('$RLDX_ROOT/'), rldx.__file__; print('[i] rldx ->', rldx.__file__, '|', sys.executable)" || exit 1
done

SPIDS=(); WPIDS=(); ALLPIDS=(); NAMES=()
kill_tree () { local c; for c in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$c"; done
               for c in $(pgrep -P "$1" 2>/dev/null); do kill -9 "$c" 2>/dev/null; done; }
cleanup () { for w in "${WPIDS[@]}"; do kill "$w" 2>/dev/null; done   # a planned stop is not a crash
             for sp in "${SPIDS[@]}"; do kill "$sp" 2>/dev/null; done
             sleep 5; kill_tree $$; for sp in "${SPIDS[@]}"; do kill -9 "$sp" 2>/dev/null; done; }
trap cleanup EXIT TERM INT

for AI in "${ARMS[@]}"; do
  EAG_CFG_W=${ARM_W[$AI]}; ARM=${ARM_NAME[$AI]}
  PORT=$((PORT_BASE + IDX * 2 + AI))
  SERVER_LOG=$(eqs_logfile "rldxclient_sail_${ARM}_server_cell${IDX}.log")
  ( cd "$RLDX_ROOT" && \
    PYTHONPATH="$RLDX_ROOT" HF_HOME="$SERVER_HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$SERVER_PY" -u "$RLDX_ROOT/rldx/eval/run_rldx_server.py" \
      --model-path "$CKPT" --embodiment-tag GENERAL_EMBODIMENT \
      --transport zmq --use-sim-policy-wrapper --eag-cfg-weight "$EAG_CFG_W" \
      --host 127.0.0.1 --port "$PORT" ) >>"$SERVER_LOG" 2>&1 &
  SPIDS+=($!)
  eqs_wait_server "${SPIDS[-1]}" "$PORT" 1800 "$SERVER_LOG" || exit 1
  grep -q "\[eag\] cfg weight" "$SERVER_LOG" \
    && echo "[i] $ARM server: $(grep '\[eag\] cfg weight' "$SERVER_LOG" | tail -n 1)" \
    || { echo "[ERR] $ARM server did not set the EAG cfg weight"; exit 1; }

  OUTBASE="$(eqs_outdir "${TAG0}_cfgw${EAG_CFG_W}_step60000_s${SEED}")"
  NOW=$(date +%s); LEFT=$(( (DEADLINE - NOW) / 60 ))
  PIDS=()
  for i in "${SEL[@]}"; do
    IFS=: read -r TASK KP <<< "${TASKCFG[$i]}"
    KP_ARG=0; [ "$KP" != "150" ] && KP_ARG=$KP
    ODIR="$OUTBASE/$ARM/$TASK"; mkdir -p "$ODIR"
    echo "[i] start $ARM/$TASK kp=$KP port=$PORT -> $ODIR (${LEFT} min left)"
    ( cd "$RLDX_ROOT" && \
      PYTHONPATH="$RLDX_ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      timeout -k 60 $(( LEFT * 60 )) \
      "$CLIENT_PY" -u "$RLDX_ROOT/rldx/eval/sail_rollout_robocasa.py" \
        --policy_client_host 127.0.0.1 --policy_client_port "$PORT" \
        --env_name "robocasa_panda_omron/${TASK}_PandaOmron_Env" \
        --n_episodes "$N_EP" --max_episode_steps "$MAX_EP_STEPS" --seed "$SEED" \
        --video_dir "$ODIR" --replan_steps "$REPLAN" \
        --eag_horizon "$EAG_HORIZON" --eag_gate "$EAG_GATE" --pos_teb "$POS_TEB" --ori_teb "$ORI_TEB" \
        --use_precision --c_slow "$C_SLOW" --c_fast "$C_FAST" \
        --precision_threshold "$PRECISION_THR" --slowdown_window_size "$SLOWDOWN_WINDOW" \
        --kp "$KP_ARG" --damping_ratio "$DAMPING" ) >>"$ODIR/eval.log" 2>&1 &
    PIDS+=($!); ALLPIDS+=($!); NAMES+=("$ARM/$TASK")
  done
  eqs_guard_clients "${SPIDS[-1]}" "${PIDS[@]}"
  WPIDS+=("$EQS_WATCHDOG_PID")
done

for i in "${!ALLPIDS[@]}"; do
  if wait "${ALLPIDS[$i]}"; then echo "[i] OK   ${NAMES[$i]}"
  else echo "[ERR] ${NAMES[$i]} rc=$? -- re-submit to continue from prediction.txt"; fi
done
for AI in "${ARMS[@]}"; do
  OUTBASE="$MODEL_OUTPUT_DIR/${TAG0}_cfgw${ARM_W[$AI]}_step60000_s${SEED}"
  for i in "${SEL[@]}"; do
    IFS=: read -r TASK _ <<< "${TASKCFG[$i]}"
    f="$OUTBASE/${ARM_NAME[$AI]}/$TASK/prediction.txt"
    echo "[i]   ${ARM_NAME[$AI]}/$TASK: $(grep -ac '^episode' "$f" 2>/dev/null || echo 0)/$N_EP ep"
  done
done
echo "[$(date '+%F %T')] cell $IDX done"
cleanup
exit 0
