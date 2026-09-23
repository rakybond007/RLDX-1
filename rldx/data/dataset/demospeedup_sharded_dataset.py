"""Sharded LIBERO loader for explicit DemoSpeedup targets and validity masks."""

import json

import numpy as np

from rldx.data.types import MessageType

from .lerobot_episode_loader import LeRobotEpisodeLoader
from .sharded_single_step_dataset import ShardedSingleStepDataset, extract_step_data


class DemoSpeedupEpisodeLoader(LeRobotEpisodeLoader):
    extra_parquet_columns = ("speedup.actions", "speedup.valid")

    def get_dataset_statistics(self) -> dict:
        stats = super().get_dataset_statistics()
        speedup = json.loads((self.dataset_path / "meta/speedup_action_stats.json").read_text())
        if speedup.get("count", 0) <= 0:
            raise ValueError("DemoSpeedup statistics contain no valid action targets")
        for key, group in self.modality_meta["action"].items():
            if key not in stats["action"]:
                continue
            lo, hi = group["start"], group["end"]
            stats["action"][key] = {
                name: speedup[name][lo:hi]
                for name in ("mean", "std", "min", "max", "q01", "q99")
            }
        return stats


class DemoSpeedupShardedSingleStepDataset(ShardedSingleStepDataset):
    def __init__(self, *args, **kwargs):
        if kwargs["modality_configs"]["action"].delta_indices != list(range(8)):
            raise ValueError("DemoSpeedup targets require action horizon 8")
        super().__init__(*args, episode_loader_class=DemoSpeedupEpisodeLoader, **kwargs)

    def get_effective_episode_length(self, episode_index: int) -> int:
        # Only the final frame has no complete 2-step target. Each earlier row
        # already contains its entire 8-step chunk, including its own padding.
        return max(0, self.episode_loader.get_episode_length(episode_index) - 1)

    def get_datapoint(self, episode_data, step_index: int) -> dict:
        assert self.processor is not None, "Processor must be set before getting datapoints"
        row = episode_data.iloc[step_index]
        targets = np.asarray(row["speedup.actions"], dtype=np.float32).reshape(8, 7)
        valid = np.asarray(row["speedup.valid"], dtype=bool)
        if valid.shape != (8,) or not valid.any() or not np.all(valid == (np.arange(8) < valid.sum())):
            raise ValueError(f"Invalid DemoSpeedup target mask at frame {step_index}")
        observation_configs = {k: v for k, v in self.modality_configs.items() if k != "action"}
        step = extract_step_data(
            episode_data, step_index, observation_configs, self.embodiment_tag, self.allow_padding
        )
        for key in self.modality_configs["action"].modality_keys:
            group = self.episode_loader.modality_meta["action"][key]
            step.actions[key] = targets[:, group["start"] : group["end"]]
        processed = self.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
        processed["action_mask"][:8][~valid] = 0
        processed["action"][:8][~valid] = 0
        return processed
