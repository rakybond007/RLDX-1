# SPDX-License-Identifier: Apache-2.0
"""LIBERO modality config with the ATQ conf label carrier.

Identical to ``libero_config.py`` except that the action modality carries one
extra key, ``ratio_label``, whose two dims hold ``[conf, valid]`` baked into the
dataset by ``run_scripts/data/atq_labels/materialize_conf_v7.py`` (action
7 -> 9, ``meta/modality.json`` gains ``ratio_label: {start: 7, end: 9}``).

``RLDXProcessor`` strips the carrier BEFORE normalisation (``conf_carrier_key``)
and re-emits it as ``conf_target`` / ``conf_valid``. Concatenated action layout:
eef_pos_delta [0,3) | eef_rot_delta [3,6) | gripper_close [6,7), so the
gripper is discrete dim ``6`` (``--atq-discrete-action-dims 6``).
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


ATQ_CONF_CARRIER_KEY = "ratio_label"

libero_conf_config = {
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
            ATQ_CONF_CARRIER_KEY,
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
            # ratio_label = [conf, valid]; ABSOLUTE so the loader never converts
            # it. Skipped by the normaliser via conf_carrier_key.
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


register_modality_config(libero_conf_config, EmbodimentTag.GENERAL_EMBODIMENT)
