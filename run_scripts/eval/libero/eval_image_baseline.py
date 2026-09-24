"""LIBERO benchmark: official initial states, suite horizons, replan five."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import random
import time

os.environ["RLDX_SKIP_HF_REGISTRATION"] = "1"

import imageio.v2 as imageio
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from rldx.eval.sim.LIBERO.libero_env import LiberoEnv
from rldx.policy.server_client import PolicyClient

HORIZONS = {'libero_spatial': 220, 'libero_object': 280, 'libero_goal': 300, 'libero_10': 520}
ACTION_KEYS = ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper')


def disable_arm_action_clip():
    """Keep the OSC affine action scale while removing its input saturation."""
    from robosuite.controllers.osc import OperationalSpaceController

    def scale_action(self, action):
        if self.action_scale is None:
            self.action_scale = abs(self.output_max - self.output_min) / abs(self.input_max - self.input_min)
            self.action_output_transform = (self.output_max + self.output_min) / 2.0
            self.action_input_transform = (self.input_max + self.input_min) / 2.0
        return (np.asarray(action, dtype=np.float64) - self.action_input_transform) * self.action_scale + self.action_output_transform

    OperationalSpaceController.scale_action = scale_action


def configure_controller(env, no_action_clip, gripper_gain_scale):
    """Check the live OSC and scale gripper actuator kp after every reset."""
    base = env._env
    for _ in range(10):
        if hasattr(base, 'robots'):
            break
        base = getattr(base, 'env', base)
    robots = getattr(base, 'robots', [])
    if not robots:
        raise RuntimeError('LIBERO robot not found after reset')
    controller = robots[0].controller
    if no_action_clip:
        command = np.zeros_like(np.asarray(controller.input_max, dtype=float))
        command[0] = 40.7
        scaled = np.asarray(controller.scale_action(command), dtype=float)
        rail = float(np.asarray(controller.output_max, dtype=float).reshape(-1)[0])
        if not np.isfinite(scaled).all() or abs(scaled.reshape(-1)[0]) <= 1.5 * abs(rail):
            raise RuntimeError(f'OSC action clip still active: probe={scaled.reshape(-1)[0]}, rail={rail}')
    if gripper_gain_scale != 1.0:
        indexes = list(getattr(robots[0], '_ref_joint_gripper_actuator_indexes', None) or [])
        if not indexes:
            raise RuntimeError('LIBERO gripper actuator indexes not found')
        model = base.sim.model
        for index in indexes:
            model.actuator_gainprm[index, 0] *= gripper_gain_scale
            model.actuator_biasprm[index, 1] *= gripper_gain_scale
        print(f'GRIPPER_GAIN {model.actuator_gainprm[indexes[0], 0]:.1f} ({len(indexes)} actuators)', flush=True)


def pack_observation(obs):
    return {k: ([v] if k.startswith('annotation.') else np.asarray(v)[None, None])
            for k, v in obs.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', choices=HORIZONS, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--replan-steps', type=int, default=5)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--task-index', type=int, default=-1)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--no-action-clip', action='store_true')
    p.add_argument('--gripper-gain-scale', type=float, default=1.0)
    a = p.parse_args()
    assert a.replan_steps == 5, 'This baseline uses replan_steps=5'
    assert 1 <= a.episodes <= 50
    assert a.gripper_gain_scale > 0
    if a.no_action_clip:
        disable_arm_action_clip()
    random.seed(a.seed)
    np.random.seed(a.seed)
    a.output.mkdir(parents=True, exist_ok=True)
    ledger = a.output / 'episodes.jsonl'
    previous = [json.loads(line) for line in ledger.read_text().splitlines()] if a.resume and ledger.exists() else []
    seen = {(row['task_id'], row['episode']) for row in previous}
    assert len(seen) == len(previous), 'Duplicate episodes in evaluation ledger'
    client = PolicyClient(host='127.0.0.1', port=a.port, timeout_ms=2000, strict=False)
    deadline = time.monotonic() + 1200
    while not client.ping():
        if time.monotonic() > deadline:
            raise RuntimeError('Policy server readiness timed out')
        time.sleep(2)
    client.timeout_ms = 120000
    client._init_socket()
    suite = benchmark.get_benchmark_dict()[a.suite]()
    results = []
    tasks = range(suite.n_tasks) if a.task_index < 0 else [a.task_index]
    for task_id in tasks:
        task = suite.get_task(task_id)
        # Official LIBERO files contain NumPy arrays; torch 2.6 changed the
        # default to weights_only=True. These are the trusted local assets.
        initial_states = torch.load(
            Path(get_libero_path('init_states')) / task.problem_folder / task.init_states_file,
            weights_only=False,
        )
        assert len(initial_states) >= a.episodes
        env = LiberoEnv(str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file), task.language)
        env._env.seed(a.seed)
        try:
            for episode in range(a.episodes):
                if (task_id, episode) in seen:
                    continue
                env.reset()
                configure_controller(env, a.no_action_clip, a.gripper_gain_scale)
                raw = env._env.set_init_state(initial_states[episode])
                for _ in range(10):
                    raw, _, _, _ = env._env.step([0.0] * 6 + [-1.0])
                obs = env._process_observation(raw)
                client.reset()
                actions = deque()
                success = False
                frames = []
                calls = 0
                for step in range(HORIZONS[a.suite]):
                    if episode == 0:
                        frames.append(obs['video.image'].copy())
                    if not actions:
                        predicted, _ = client.get_action(pack_observation(obs))
                        for k in ACTION_KEYS:
                            value = predicted['action.' + k]
                            assert value.shape[0] == 1 and value.shape[1] >= a.replan_steps
                            assert np.isfinite(value).all(), k
                        for t in range(a.replan_steps):
                            actions.append({'action.' + k: predicted['action.' + k][0, t].copy() for k in ACTION_KEYS})
                        calls += 1
                    obs, _, done, truncated, info = env.step(actions.popleft())
                    success = bool(info.get('success', False))
                    if success or done or truncated:
                        break
                result = dict(suite=a.suite, task_id=task_id, task=task.name, episode=episode,
                              initial_state_id=episode, success=success, steps=step + 1,
                              policy_calls=calls, replan_steps=a.replan_steps, seed=a.seed,
                              no_action_clip=a.no_action_clip, gripper_gain_scale=a.gripper_gain_scale)
                results.append(result)
                with ledger.open('a') as f:
                    f.write(json.dumps(result) + '\n')
                print('EVAL_EPISODE ' + json.dumps(result), flush=True)
                if frames:
                    imageio.mimsave(a.output / f'{task_id:02d}_episode0_{success}.mp4', frames, fps=20)
        finally:
            env.close()
    all_results = previous + results
    summary = dict(suite=a.suite, episodes=len(all_results), successes=sum(r['success'] for r in all_results),
                   success_rate=sum(r['success'] for r in all_results) / len(all_results), replan_steps=a.replan_steps,
                   seed=a.seed, max_episode_steps=HORIZONS[a.suite], settling_steps=10,
                   no_action_clip=a.no_action_clip, gripper_gain_scale=a.gripper_gain_scale)
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('EVAL_SUMMARY ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
