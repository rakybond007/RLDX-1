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

"""RoboCasa modality config for the SAIL reproduction (EAG + precision bit).

Identical to `robocasa_config.py` except for the trailing `precision` action
column. `sail_robocasa_delta_lerobot`'s `meta/modality.json` declares the same
video/state/action key names, plus `precision` at columns 12:13 — so the
baseline config carries over and the action dimension goes 12 -> 13.

`precision` is a 0/1 label, normalized min/max like every other key
(`ModalityConfig`'s default), and thresholded at eval by `precision_threshold`.

Load it in place of `robocasa_config.py`:

    --modality-config-path rldx/configs/data/robocasa_sail_config.py

Never import both in one process: `register_modality_config` asserts that a
tag is registered only once, and both files claim GENERAL_EMBODIMENT.
"""

import copy

from rldx.configs.data.embodiment_configs import register_modality_config
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# Based on `robocasa_panda_omron` configs in: https://huggingface.co/nvidia/GR00T-N1.6-3B/blob/main/processor_config.json


robocasa_sail_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_view", "right_view", "wrist_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "end_effector_position_relative",
            "end_effector_rotation_relative",
            "gripper_qpos",
            "base_position",
            "base_rotation",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=[
            "end_effector_position",
            "end_effector_rotation",
            "gripper_close",
            "base_motion",
            "control_mode",
            "precision",
        ],
        action_configs=[
            # end_effector_position
            ActionConfig(
                rep=ActionRepresentation.DELTA,
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
            # end_effector_rotation
            ActionConfig(
                rep=ActionRepresentation.DELTA,
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
            # gripper_close
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # base_motion
            ActionConfig(
                rep=ActionRepresentation.DELTA,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # control_mode
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # precision — SAIL 0/1 label, not a commanded motion.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}


register_modality_config(robocasa_sail_config, EmbodimentTag.GENERAL_EMBODIMENT)
register_modality_config(copy.deepcopy(robocasa_sail_config), EmbodimentTag.NEW_EMBODIMENT)
