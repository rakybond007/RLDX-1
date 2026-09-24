"""RoboCasa image config with the eight-target DemoSpeedup action horizon."""

from rldx.configs.data.robocasa_config import robocasa_panda_omron


robocasa_panda_omron["action"].delta_indices = list(range(8))
