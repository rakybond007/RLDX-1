# SPDX-License-Identifier: Apache-2.0
"""RoboCasa modality config with the ATQ conf label carrier.

Identical to ``robocasa_config.py`` except that the action modality carries one
extra key, ``ratio_label``, whose two dims hold ``[conf, valid]`` baked into the
dataset by ``run_scripts/data/atq_labels/materialize_conf_dataset.py`` (action
12 -> 14, ``meta/modality.json`` gains ``ratio_label: {start: 12, end: 14}``).

The carrier has to be listed here so the LeRobot loader slices it out of the
flat ``action`` array; ``RLDXProcessor`` then strips it BEFORE normalisation
(``conf_carrier_key``) and re-emits it as ``conf_target`` / ``conf_valid`` for
the ATQ conf head. It never reaches the concatenated action tensor, the loss
mask, or ``decode_action``.

Use with::

    --use-atq-moe --atq-discrete-action-dims 6 11 \
    --modality-config-path rldx/configs/data/robocasa_conf_config.py

``6`` = gripper_close, ``11`` = control_mode in the concatenated 12-dim action
(pos 3 | rot 3 | gripper 1 | base_motion 4 | control_mode 1). Verify against the
dataset's ``meta/modality.json`` before training.
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


ATQ_CONF_CARRIER_KEY = "ratio_label"

robocasa_panda_omron_conf = {
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
            ATQ_CONF_CARRIER_KEY,
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
            # ratio_label = [conf, valid]; ABSOLUTE so the loader never zero-pads
            # or converts it. Skipped by the normaliser via conf_carrier_key.
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


register_modality_config(robocasa_panda_omron_conf, EmbodimentTag.GENERAL_EMBODIMENT)
register_modality_config(copy.deepcopy(robocasa_panda_omron_conf), EmbodimentTag.NEW_EMBODIMENT)
