# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified from the original NVIDIA Isaac GR00T N1.7.
# Original source: https://github.com/NVIDIA/Isaac-GR00T

"""
LIBERO environment

This file wraps the original LIBERO as a Gymnasium environment,
and registers it so that it can be instantiated via gym.make(...) and work
using our distributed evaluation.
"""

import math
import os

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from libero.libero import benchmark


os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils import get_libero_path
import numpy as np


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action


# Per-task OSC position gain (Table 8). Keyed by task name, not index: the
# installed LIBERO build is LIBERO-plus, whose suites hold ~2500 augmented
# variants each, so a positional index would address the wrong task entirely.
# The 40 names below are the original LIBERO tasks the eval runner iterates.
# LIBERO's own default is 150 everywhere; robosuite applies kp unclamped in the
# "fixed" impedance mode, so the 600 entry survives kp_limits of [0, 300].
# Opt-in: set LIBERO_OSC_KP=table8 to apply these (one process per task, so a
# default-on switch would change an already-running eval midway).
OSC_KP = {
    # libero_spatial
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate": 150,
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate": 150,
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate": 300,
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate": 300,
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate": 150,
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate": 150,
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate": 300,
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate": 150,
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate": 150,
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate": 150,
    # libero_object
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket": 300,
    "pick_up_the_cream_cheese_and_place_it_in_the_basket": 150,
    "pick_up_the_salad_dressing_and_place_it_in_the_basket": 300,
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket": 150,
    "pick_up_the_ketchup_and_place_it_in_the_basket": 300,
    "pick_up_the_tomato_sauce_and_place_it_in_the_basket": 300,
    "pick_up_the_butter_and_place_it_in_the_basket": 300,
    "pick_up_the_milk_and_place_it_in_the_basket": 300,
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket": 600,
    "pick_up_the_orange_juice_and_place_it_in_the_basket": 300,
    # libero_goal
    "open_the_middle_drawer_of_the_cabinet": 300,
    "put_the_bowl_on_the_stove": 150,
    "put_the_wine_bottle_on_top_of_the_cabinet": 150,
    "open_the_top_drawer_and_put_the_bowl_inside": 150,
    "put_the_bowl_on_top_of_the_cabinet": 150,
    "push_the_plate_to_the_front_of_the_stove": 150,
    "put_the_cream_cheese_in_the_bowl": 150,
    "turn_on_the_stove": 150,
    "put_the_bowl_on_the_plate": 150,
    "put_the_wine_bottle_on_the_rack": 300,
    # libero_10
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket": 300,
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket": 150,
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it": 150,
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it": 300,
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": 150,
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy": 300,
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate": 150,
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket": 300,
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": 150,
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it": 150,
}


class LiberoEnv(gym.Env):
    """LanguageTable env."""

    def __init__(self, task_bddl_file: str, task_description: str, osc_kp: int | None = None):
        self._env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=256,
            camera_widths=256,
        )
        if osc_kp is not None:
            # This LIBERO build takes a controller *name* and loads the config
            # itself, so there is no controller_configs argument to override.
            # The gain has to go into ``robot_configs``: LIBERO runs with
            # hard_reset, and ``_load_robots`` rebuilds every Robot from that
            # dict on each reset, discarding anything set on the robot or the
            # controller. The live controller is patched too, for the episode
            # that runs before the first reset.
            sim_env = self._env.env
            for cfg in sim_env.robot_configs:
                cfg["controller_config"]["kp"] = osc_kp
            for robot in sim_env.robots:
                robot.controller_config["kp"] = osc_kp
                ctrl = robot.controller
                ctrl.kp = ctrl.nums2array(osc_kp, 6)
                ctrl.kd = 2 * np.sqrt(ctrl.kp)  # damping_ratio is 1 in the stock config
        self._task_description = task_description
        # Convert Gym action space to Gymnasium.
        self.observation_space = gym.spaces.Dict(
            {
                "video.image": gym.spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
                "video.wrist_image": gym.spaces.Box(
                    low=0, high=255, shape=(256, 256, 3), dtype=np.uint8
                ),
                "state.x": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.y": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.z": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.roll": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.pitch": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.yaw": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.gripper": gym.spaces.Box(low=-1, high=1, shape=(2,)),
                "annotation.human.action.task_description": gym.spaces.Text(max_length=512),
            }
        )
        self.action_space = spaces.Dict(
            {
                "action.x": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.y": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.z": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.roll": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.pitch": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.yaw": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.gripper": spaces.Box(low=-1, high=1, shape=(1,)),
            }
        )

    def close(self):
        self._env.close()

    def _process_observation(self, obs):
        xyz = obs["robot0_eef_pos"]
        rpy = quat2axisangle(obs["robot0_eef_quat"])
        gripper = obs["robot0_gripper_qpos"]
        new_obs = {
            "video.image": obs["agentview_image"][::-1, ::-1],
            "video.wrist_image": obs["robot0_eye_in_hand_image"][::-1, ::-1],
            "state.x": [xyz[0]],
            "state.y": [xyz[1]],
            "state.z": [xyz[2]],
            "state.roll": [rpy[0]],
            "state.pitch": [rpy[1]],
            "state.yaw": [rpy[2]],
            "state.gripper": gripper,
            "annotation.human.action.task_description": self._task_description,
        }
        return new_obs

    def reset(self, seed=None, options=None):
        observation = self._env.reset()
        observation = self._process_observation(observation)
        info = {"success": self._env.check_success()}
        return observation, info

    def step(self, action):
        action_vector = np.concatenate(
            [
                action["action.x"],
                action["action.y"],
                action["action.z"],
                action["action.roll"],
                action["action.pitch"],
                action["action.yaw"],
                action["action.gripper"],
            ],
            axis=0,
        )
        action_vector = normalize_gripper_action(action_vector)
        action_vector = invert_gripper_action(action_vector)
        observation, reward, done, info = self._env.step(action_vector)
        observation = self._process_observation(observation)
        info["success"] = self._env.check_success()
        # LIBERO's bddl_base_domain.step replaces robosuite's ``done`` with
        # ``_check_success()``, so a horizon-terminated episode looks unfinished
        # from here while robosuite refuses any further step. Surface that as
        # truncation, otherwise the caller keeps stepping and robosuite raises
        # "executing action in terminated episode".
        truncated = bool(getattr(self._env.env, "done", False))
        return observation, reward, done, truncated, info


def register_libero_envs():
    benchmark_dict = benchmark.get_benchmark_dict()
    for task_suite_name in [
        "libero_10",
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_90",
    ]:
        task_suite = benchmark_dict[task_suite_name]()
        for task_id in range(task_suite.get_num_tasks()):
            task = task_suite.get_task(task_id)
            task_name = task.name
            task_description = task.language
            task_bddl_file = os.path.join(
                get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
            )
            kwargs = {
                "task_bddl_file": task_bddl_file,
                "task_description": task_description,
            }
            if os.environ.get("LIBERO_OSC_KP", "default") == "table8":
                if task_name in OSC_KP:
                    kwargs["osc_kp"] = OSC_KP[task_name]
            register(
                id=f"libero_sim/{task_name}",
                entry_point="rldx.eval.sim.LIBERO.libero_env:LiberoEnv",
                kwargs=kwargs,
            )


if __name__ == "__main__":
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_10"  # can also choose libero_spatial, libero_object, etc.
    task_suite = benchmark_dict[task_suite_name]()
    for key in ["libero_10", "libero_spatial", "libero_object", "libero_goal", "libero_90"]:
        for task_name in benchmark_dict[key]().get_task_names():
            print(f"- {key}/{task_name}")

    # retrieve a specific task
    task_id = 0
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    print(
        f"[info] retrieving task {task_id} from suite {task_suite_name}, the "
        + f"language instruction is {task_description}, and the bddl file is {task_bddl_file}"
    )

    # step over the environment
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 128, "camera_widths": 128}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()
    init_states = task_suite.get_task_init_states(
        task_id
    )  # for benchmarking purpose, we fix the a set of initial states
    init_state_id = 0
    env.set_init_state(init_states[init_state_id])

    dummy_action = [0.0] * 7
    for step in range(10):
        obs, reward, done, info = env.step(dummy_action)
        print("step", step, "obs", obs.keys())
    env.close()
