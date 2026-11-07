# SPDX-License-Identifier: Apache-2.0
"""RoboCasa SAIL rollout on the RLDX client.

The environment is built exactly like `rldx/eval/rollout_policy.py` builds it
(`create_eval_env`: `gym.make("robocasa_panda_omron/<task>_PandaOmron_Env")`,
`VideoRecordingWrapper`, `MultiStepWrapper`), so the scene is RoboCasa's own eval
scene (object split B, original textures, 5 layout/style pairs) and the images go
through the same processing as every other RLDX RoboCasa eval.

What this file adds on top is SAIL's execution loop, ported from the Isaac-GR00T
client `scripts/robocasa_service_sail_delta.py` (same arithmetic):

  replan_steps   rows of each 16-row chunk that are executed before re-inferring (12).
  EAG guide      the next `eag_horizon` rows of the previous chunk (physical units),
                 sent as `options["eag_guide"]`; with `eag_gate=goal` only while the arm
                 controller's tracking error is inside `pos_teb` / `ori_teb`.
  precision      the chunk's 13th column; a window of `slowdown_window_size` rows around
                 the current row decides `c_slow` vs `c_fast`, and the env step runs at
                 `control_freq = 20 / c`.
  kp             OSC arm gain (kd = 2 sqrt(kp) damping_ratio), re-applied after every reset.

Per episode it appends to `<video_dir>/prediction.txt`:
  episode N is_success: [ True] action_steps: A sim_time: S precise_rate: P n_infer: I eag_guided: g/c
`action_steps` = env steps, `sim_time` = MuJoCo time actually simulated [s]. A re-run
continues after the last recorded episode (the per-episode reset seed is
base_seed * 100000 + episode_id, so a resumed episode sees the same scene).
"""

import argparse
import collections
from pathlib import Path
import re
import sys
import time
import uuid

import numpy as np

from rldx.eval.rollout_policy import (
    MultiStepConfig,
    VideoConfig,
    WrapperConfigs,
    create_eval_env,
)


ACTION_KEYS = (
    ("action.end_effector_position", 3),
    ("action.end_effector_rotation", 3),
    ("action.gripper_close", 1),
    ("action.base_motion", 4),
    ("action.control_mode", 1),
    ("action.precision", 1),
)
ACTION_DIM = sum(d for _, d in ACTION_KEYS)  # 13
PRECISION_COL = 12
GRIPPER_COL = 6
CONTROL_MODE_COL = 11
BINARY_COLS = (GRIPPER_COL, CONTROL_MODE_COL)  # 0/1 labels; the model emits them continuous
CHUNK_HORIZON = 16
BASE_CONTROL_FREQ = 20.0
_EP_RE = re.compile(r"^episode\s+(\d+)\s+is_success:")


# ------------------------------------------------------------------ robosuite handles
def _robosuite_env(env):
    """MultiStepWrapper -> ... -> GrootRoboCasaEnv -> robosuite Kitchen."""
    return env.unwrapped.env


def _restore_base_control_freq(base, base_cf):
    if float(base.control_freq) != float(base_cf):
        base.control_freq = float(base_cf)
        base.control_timestep = 1.0 / float(base_cf)


def _set_control_freq(base, cf):
    base.control_freq = float(cf)
    base.control_timestep = 1.0 / float(cf)


def _arm_parts(base):
    parts = getattr(base.robots[0].composite_controller, "part_controllers", None)
    if not parts:
        raise SystemExit("composite_controller.part_controllers not found")
    return {n: c for n, c in parts.items() if any(t in n for t in ("right", "left", "arm"))}


def _apply_gains(base, kp, damping_ratio):
    """kp <= 0 keeps the env default. reset() rebuilds the controllers, so call it per episode."""
    if kp <= 0:
        return []
    touched = []
    for name, ctrl in _arm_parts(base).items():
        if not hasattr(ctrl, "kp"):
            continue
        ctrl.kp = np.full_like(np.asarray(ctrl.kp, dtype=np.float64), float(kp))
        ctrl.kd = 2.0 * np.sqrt(ctrl.kp) * float(damping_ratio)
        touched.append(name)
    if not touched:
        raise SystemExit("no arm part controller to set kp on")
    return touched


def _arm_controller(base):
    for ctrl in _arm_parts(base).values():
        if hasattr(ctrl, "ref_pos") or hasattr(ctrl, "ee_pos"):
            return ctrl
    raise SystemExit("no arm part controller with a measured pose")


def _ctrl_measured_pose(ctrl):
    """Measured pose in the frame the controller holds its goal in."""
    if hasattr(ctrl, "ref_pos"):
        if getattr(ctrl, "input_ref_frame", "world") == "base":
            return (np.asarray(ctrl.world_to_origin_frame(ctrl.ref_pos), dtype=np.float64),
                    np.asarray(ctrl.goal_origin_to_eef_pose()[:3, :3], dtype=np.float64))
        return (np.asarray(ctrl.ref_pos, dtype=np.float64),
                np.asarray(ctrl.ref_ori_mat, dtype=np.float64))
    return (np.asarray(ctrl.ee_pos, dtype=np.float64),
            np.asarray(ctrl.ee_ori_mat, dtype=np.float64))


def _rot_angle_between(r_a, r_b):
    trace = np.trace(np.asarray(r_a, dtype=np.float64).T @ np.asarray(r_b, dtype=np.float64))
    return float(np.arccos(np.clip((np.clip(trace, -1.0, 3.0) - 1.0) / 2.0, -1.0, 1.0)))


def _eag_tracking_error(base):
    """SAIL Algorithm 2 tracking error: controller goal vs measured pose (pos inf-norm, rot angle)."""
    ctrl = _arm_controller(base)
    if getattr(ctrl, "goal_pos", None) is None or getattr(ctrl, "goal_ori", None) is None:
        return None, None
    meas_pos, meas_ori = _ctrl_measured_pose(ctrl)
    pos_err = float(np.linalg.norm(np.asarray(ctrl.goal_pos, dtype=np.float64) - meas_pos, ord=np.inf))
    return pos_err, _rot_angle_between(ctrl.goal_ori, meas_ori)


# ------------------------------------------------------------------ policy I/O
def _batch_obs(obs):
    """Single-env MultiStepWrapper obs -> the (B=1, T, ...) form the vector env hands the policy."""
    out = {}
    for k, v in obs.items():
        if isinstance(v, str):
            out[k] = [v]
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], str):
            out[k] = [v[0]]
        else:
            out[k] = np.asarray(v)[None]
    return out


def _chunk_from_action(action):
    cols = []
    for key, d in ACTION_KEYS:
        if key not in action:
            raise SystemExit(f"server response has no {key}: {sorted(action)}")
        v = np.asarray(action[key], dtype=np.float64)
        if v.ndim == 3:
            v = v[0]
        v = v.reshape(v.shape[0], -1)
        if v.shape[-1] != d:
            raise SystemExit(f"{key} width {v.shape[-1]} != {d}")
        cols.append(v)
    chunk = np.concatenate(cols, axis=-1)
    # Same 0/1 read-out as the Isaac-GR00T adapter (`BINARY_THRESHOLD` 0.5), so the guide
    # carries the discrete values the env acts on.
    for c in BINARY_COLS:
        chunk[:, c] = (chunk[:, c] >= 0.5).astype(np.float64)
    return chunk


def _env_action(row):
    """(13,) row -> MultiStepWrapper(n_action_steps=1) action dict; precision stays out of the env."""
    out, lo = {}, 0
    for key, d in ACTION_KEYS:
        if key != "action.precision":
            out[key] = np.asarray(row[lo:lo + d], dtype=np.float64).reshape(1, d)
        lo += d
    return out


def _slow_decision(args, step_row, plan_ahead, exec_prec):
    """Same window as the Isaac-GR00T client (`speed_mode=precision`)."""
    p_now = float(step_row[PRECISION_COL])
    if args.slowdown_window_size <= 1:
        return p_now >= args.precision_threshold
    seg = args.slowdown_window_size // 2
    left = exec_prec[-min(len(exec_prec), seg):]
    right = [p_now] + [float(a[PRECISION_COL]) for a in plan_ahead[:max(0, seg - 1)]]
    return any(v >= args.precision_threshold for v in list(left) + [p_now] + right)


def _load_done(pred_path):
    done = {}
    if pred_path.exists():
        for line in open(pred_path, errors="ignore"):
            m = _EP_RE.match(line.strip())
            if m:
                done[int(m.group(1))] = line.strip()
    return done


# ------------------------------------------------------------------ main
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--policy_client_host", type=str, default="127.0.0.1")
    p.add_argument("--policy_client_port", type=int, required=True)
    p.add_argument("--env_name", type=str, required=True,
                   help="robocasa_panda_omron/<Task>_PandaOmron_Env")
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--max_episode_steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--video_dir", type=str, required=True)
    p.add_argument("--replan_steps", type=int, default=12)
    p.add_argument("--no_eag", dest="eag", action="store_false", default=True)
    p.add_argument("--eag_horizon", type=int, default=4)
    p.add_argument("--eag_gate", type=str, default="goal", choices=("open", "goal"))
    p.add_argument("--pos_teb", type=float, default=0.02)
    p.add_argument("--ori_teb", type=float, default=0.05)
    p.add_argument("--use_precision", action="store_true")
    p.add_argument("--c_slow", type=float, default=1.0)
    p.add_argument("--c_fast", type=float, default=1.0)
    p.add_argument("--precision_threshold", type=float, default=0.5)
    p.add_argument("--slowdown_window_size", type=int, default=6)
    p.add_argument("--kp", type=float, default=0.0, help="0 keeps the env default (150)")
    p.add_argument("--damping_ratio", type=float, default=1.0)
    return p


def main():
    args = build_parser().parse_args()
    if not 0 < args.replan_steps <= CHUNK_HORIZON:
        raise SystemExit(f"--replan_steps must be in 1..{CHUNK_HORIZON}")
    if args.eag and args.replan_steps + args.eag_horizon > CHUNK_HORIZON:
        raise SystemExit("--replan_steps + --eag_horizon exceeds the chunk")
    for name, c in (("c_slow", args.c_slow), ("c_fast", args.c_fast)):
        if not 0.0 < c <= 1.0:
            raise SystemExit(f"--{name} must be in (0, 1]: {c}")

    out_dir = Path(args.video_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "prediction.txt"
    done = _load_done(pred_path)
    n_done = (max(done) + 1) if done else 0
    if n_done >= args.n_episodes:
        print(f"[sail-rldx] {args.env_name}: {n_done}/{args.n_episodes} already recorded", flush=True)
        return

    from rldx.policy.server_client import PolicyClient

    policy = PolicyClient(host=args.policy_client_host, port=args.policy_client_port)
    mc = policy.get_modality_config()
    wrapper_configs = WrapperConfigs(
        video=VideoConfig(video_dir=str(out_dir), max_episode_steps=args.max_episode_steps),
        multistep=MultiStepConfig(
            video_delta_indices=np.array(mc["video"].delta_indices),
            state_delta_indices=np.array(mc["state"].delta_indices) if "state" in mc else None,
            n_action_steps=1,
            max_episode_steps=args.max_episode_steps,
            terminate_on_success=True,
        ),
    )
    # start_episode_id = n_done keeps the per-episode reset seed of a resumed episode.
    env = create_eval_env(args.env_name, env_idx=0, total_n_envs=1,
                          wrapper_configs=wrapper_configs, start_episode_id=n_done,
                          seed=args.seed)
    base = _robosuite_env(env)
    base_cf = float(base.control_freq)
    print(f"[sail-rldx] env={args.env_name} replan={args.replan_steps} eag={args.eag} "
          f"eag_horizon={args.eag_horizon} gate={args.eag_gate} use_precision={args.use_precision} "
          f"c_slow={args.c_slow} c_fast={args.c_fast} kp={args.kp} base_cf={base_cf} "
          f"n_done={n_done}", flush=True)

    session_id = f"{args.env_name}_{uuid.uuid4().hex[:8]}"
    infer_times = []
    announced = False
    for ep in range(n_done, args.n_episodes):
        _restore_base_control_freq(base, base_cf)
        obs, info = env.reset()
        touched = _apply_gains(base, args.kp, args.damping_ratio)
        if not announced:
            ctrl = _arm_controller(base)
            print(f"[arm] kp={np.asarray(ctrl.kp).reshape(-1)[:3].tolist()} "
                  f"kd={np.round(np.asarray(ctrl.kd).reshape(-1)[:3], 3).tolist()} parts={touched}",
                  flush=True)
            announced = True

        t = n_infer = n_precise = n_slow = 0
        sim_time = 0.0
        plan = collections.deque()
        exec_prec = []
        prev_chunk = None
        first = True
        eag_cand = eag_sent = 0
        success = False
        finished = False
        while not finished and t < args.max_episode_steps:
            if not plan:
                options = {"reset_memory": [first], "session_ids": [session_id]}
                if args.eag and prev_chunk is not None:
                    pos_err, ori_err = _eag_tracking_error(base)
                    do_guide = True if args.eag_gate == "open" else (
                        pos_err is not None and pos_err < args.pos_teb and ori_err < args.ori_teb)
                    eag_cand += 1
                    if do_guide:
                        guide = prev_chunk[args.replan_steps:args.replan_steps + args.eag_horizon]
                        options["eag_guide"] = np.ascontiguousarray(guide, dtype=np.float32)
                        eag_sent += 1
                t0 = time.perf_counter()
                action, _ = policy.get_action(_batch_obs(obs), options=options)
                infer_times.append(time.perf_counter() - t0)
                first = False
                chunk = _chunk_from_action(action)
                if len(chunk) < args.replan_steps:
                    raise SystemExit(f"chunk {len(chunk)} rows < --replan_steps {args.replan_steps}")
                plan.extend(chunk[:args.replan_steps])
                prev_chunk = chunk
                n_infer += 1

            row = plan.popleft()
            exec_prec.append(float(row[PRECISION_COL]))
            slow = _slow_decision(args, row, list(plan), exec_prec)
            c = (args.c_slow if slow else args.c_fast) if args.use_precision else args.c_fast
            n_slow += int(args.use_precision and slow)
            n_precise += int(row[PRECISION_COL] >= args.precision_threshold)
            cf = base_cf / float(c)
            if abs(cf - base_cf) < 1e-9:
                _restore_base_control_freq(base, base_cf)
            else:
                _set_control_freq(base, cf)
            t_before = float(base.sim.data.time)
            obs, reward, terminated, truncated, info = env.step(_env_action(row))
            sim_time += float(base.sim.data.time) - t_before
            t += 1
            s = info.get("success", False)
            success = success or bool(np.any(np.asarray(s)))
            finished = bool(terminated or truncated)

        prec_rate = (n_precise / t) if t else 0.0
        line = (f"episode {ep} is_success: [{' True' if success else 'False'}] "
                f"action_steps: {t} sim_time: {sim_time:.4f} precise_rate: {prec_rate:.4f} "
                f"n_infer: {n_infer} eag_guided: {eag_sent}/{eag_cand}")
        with open(pred_path, "a") as f:
            f.write(line + "\n")
        print(f"  {line} slow_rate={(n_slow / t) if t else 0.0:.3f}", flush=True)

    _restore_base_control_freq(base, base_cf)
    env.reset()  # closes the last episode's video file with its success/failure name
    env.close()
    recs = list(_load_done(pred_path).values())
    ok = [r for r in recs if "True" in r.split("action_steps")[0]]
    steps = [int(re.search(r"action_steps: (\d+)", r).group(1)) for r in ok]
    with open(out_dir / "results.txt", "w") as f:
        f.write(f"Total success rate: {len(ok) / len(recs) if recs else 0.0}\n")
        f.write(f"Total episodes: {len(recs)}\n")
        f.write(f"Success-only mean action_steps: {np.mean(steps) if steps else 0.0:.2f} (over {len(ok)} ep)\n")
        f.write("sail_client: rldx/eval/sail_rollout_robocasa.py (RLDX env pipeline)\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"sail_{k}: {v}\n")
        if infer_times:
            f.write(f"sail_infer_s_median: {float(np.median(infer_times)):.4f}\n")


if __name__ == "__main__":
    import logging
    import os

    rc = 0
    try:
        main()
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f"[FATAL] {e.code}", file=sys.stderr)
            rc = e.code if isinstance(e.code, int) else 1
    # MuJoCo/EGL + zmq threads keep the interpreter (and the GPU) alive past the eval.
    sys.stdout.flush()
    sys.stderr.flush()
    logging.shutdown()
    os._exit(rc)
