"""Validate episode artifacts and print missing array indices; never submits jobs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def validate_episode(path, episode, length):
    data = pd.read_parquet(path)
    if data.frame.tolist() != list(range(length)) or not (data.episode == episode).all():
        raise ValueError(f'Invalid episode/frame coverage: {path}')
    columns = ['kde_bw1', 'kde_bw1_agg', 'std_mean', 'std_mean_agg', 'entropy_z']
    if not np.isfinite(data[columns]).all().all():
        raise ValueError(f'Non-finite entropy: {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-path', default='/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta', type=Path)
    p.add_argument('--output', default='outputs/demospeedup/libero_60000_slow2_fast4', type=Path)
    p.add_argument('--num-shards', default=32, type=int)
    a = p.parse_args()
    assert a.num_shards > 0
    episodes = [json.loads(line) for line in (a.dataset_path / 'meta/episodes.jsonl').read_text().splitlines()]
    missing, done, pending_frames = set(), 0, 0
    shard_frames = [0] * a.num_shards
    for i, ep in enumerate(episodes):
        assert ep['episode_index'] == i
        shard = i % a.num_shards
        shard_frames[shard] += ep['length']
        path = a.output / f'entropy_ep{i:06d}.parquet'
        if path.exists():
            validate_episode(path, i, ep['length'])
            done += 1
        else:
            missing.add(shard)
            pending_frames += ep['length']
    print(json.dumps(dict(episodes=len(episodes), completed=done, pending_frames=pending_frames,
                         shard_frames_min=min(shard_frames), shard_frames_max=max(shard_frames),
                         pending_array_indices=sorted(missing)), indent=2))
    if missing:
        indices = ','.join(map(str, sorted(missing)))
        print('Not submitted. From repository root, after approval:')
        print('MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/checkpoints/rldx1_image_baselines '
              f'sbatch --array={indices}%2 run_scripts/demospeedup/collect_libero.sbatch')


if __name__ == '__main__':
    main()
