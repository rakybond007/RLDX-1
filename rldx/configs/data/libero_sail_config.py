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

"""LIBERO modality config for the SAIL reproduction (EAG + precision bit).

Identical to `libero_config.py` except for the trailing `precision` action
column. `sail_libero_delta_lerobot`'s `meta/modality.json` declares the same
video/state/action key names as `libero_gr00t_delta`, plus `precision` at
columns 7:8 — so the baseline config carries over unchanged and the action
dimension goes 7 -> 8.

`precision` is a 0/1 label. It is normalized like every other key (min/max is
`ModalityConfig`'s default), so the model emits a continuous value and the
evaluator thresholds it with `precision_threshold`.

Load it in place of `libero_config.py`:

    --modality-config-path rldx/configs/data/libero_sail_config.py

Never import both in one process: `register_modality_config` asserts that a
tag is registered only once, and both files claim GENERAL_EMBODIMENT.
"""

from rldx.configs.data.embodiment_configs import register_modality_config
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


libero_sail_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front_view", "left_wrist_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "eef_pos_absolute",
            "eef_rot_absolute",
            "gripper_close",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=[
            "eef_pos_delta",
            "eef_rot_delta",
            "gripper_close",
            "precision",
        ],
        action_configs=[
            # eef_pos_delta
            ActionConfig(
                rep=ActionRepresentation.DELTA,
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
            # eef_rot_delta
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


register_modality_config(libero_sail_config, EmbodimentTag.GENERAL_EMBODIMENT)
