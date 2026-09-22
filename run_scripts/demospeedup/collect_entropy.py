"""Collect every LIBERO frame using one backbone pass and ten head samples.

Outputs are normalized real-action-space entropies, not padded-dimension
entropies. The slow/fast ratios are retained for subsequent labeling/training;
collection itself never drops frames or modifies source demonstrations.
"""
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from transformers import BatchFeature
from rldx.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from rldx.data.dataset.sharded_single_step_dataset import extract_step_data
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.policy.rldx_policy import RLDXPolicy
from entropy import CausalEntropy
from plan_resume import validate_episode


@torch.inference_mode()
def verify_sampling_path(policy, collated):
    # Same RNG and observations: caching the backbone must preserve inference.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(173)
        cached = sample_chunks(policy, collated, 1)[:, 0]
        torch.manual_seed(173)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            direct = policy.model.get_action(**collated)['action_pred'].float().cpu().numpy()
    np.testing.assert_allclose(cached, direct, rtol=1e-5, atol=1e-5)
    print('ENTROPY_INFERENCE_CHECK cached_backbone_matches_full_model', flush=True)


@torch.inference_mode()
def sample_chunks(policy, collated, n_samples):
    model = policy.model
    with torch.autocast('cuda', dtype=torch.bfloat16):
        backbone_input, action_input = model.prepare_input(collated['inputs'])
        backbone_output = model.backbone(backbone_input)
        batch = backbone_output['backbone_features'].shape[0]
        def repeat(features):
            out = {}
            for key, value in features.items():
                if torch.is_tensor(value) and value.ndim > 0:
                    if value.shape[0] != batch:
                        raise ValueError(f'Unexpected batch dimension: {key}, {value.shape}')
                    value = value.repeat_interleave(n_samples, dim=0)
                out[key] = value
            return BatchFeature(data=out)
        pred = model.action_model.get_action(repeat(backbone_output), repeat(action_input))['action_pred']
    return pred.reshape(batch, n_samples, *pred.shape[1:]).float().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', required=True)
    p.add_argument('--dataset-path', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--n-samples', type=int, default=10)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=32)
    p.add_argument('--slow', type=int, default=2)
    p.add_argument('--fast', type=int, default=4)
    p.add_argument('--seed', type=int, default=7)
    a = p.parse_args()
    assert (a.slow, a.fast) == (2, 4) and a.n_samples == 10
    assert a.batch_size > 0 and 0 <= a.shard_index < a.num_shards
    a.output.mkdir(parents=True, exist_ok=True)
    # Protect against accidentally submitting the same shard twice.
    shard_lock = (a.output / f'shard_{a.shard_index}.lock').open('a')
    fcntl.flock(shard_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    policy = RLDXPolicy(model_path=a.model_path, embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
                        device='cuda', strict=False, rtc_inference_mode='none')
    for flag in ('use_memory', 'use_motion', 'use_physics'):
        assert not getattr(policy.model.config, flag, False), flag
    assert policy.model.config.video_length == 1
    assert not policy.model.training and not policy.processor.training
    config = {k: v for k, v in policy.modality_configs.items() if k != 'action'}
    assert all(v.delta_indices == [0] for k, v in config.items() if k in ('video', 'state'))
    loader = LeRobotEpisodeLoader(a.dataset_path, config, video_backend_kwargs={'num_ffmpeg_threads': 1})
    params = policy.processor.state_action_processor.norm_params['general_embodiment']['action']
    real_dim = sum(int(params[k]['dim']) for k in policy.modality_configs['action'].modality_keys)
    assert real_dim == 7, f'This LIBERO recipe expects 7 real action dimensions, got {real_dim}'
    horizon = policy.model.config.action_horizon
    episodes = list(range(a.shard_index, len(loader), a.num_shards))
    total_frames = sum(loader.get_episode_length(ep) for ep in episodes)
    manifest = dict(vars(a), output=str(a.output), real_action_dims=real_dim, action_horizon=horizon,
                    observation_stride=1, kde_bandwidth=1, temporal_aggregation=True,
                    normalized_action_space=True, episodes=len(episodes), total_frames=total_frames)
    meta = a.output / f'manifest_{a.shard_index}.json'
    if meta.exists() and json.loads(meta.read_text()) != manifest:
        raise ValueError('Existing collection has different settings; choose another output directory')
    meta.write_text(json.dumps(manifest, indent=2) + '\n')
    print('ENTROPY_MANIFEST ' + json.dumps(manifest), flush=True)
    measured_frames, measured_seconds, remaining_frames = 0, 0.0, total_frames
    verified = False
    with ThreadPoolExecutor(max_workers=1) as pool:
        for ep in episodes:
            target = a.output / f'entropy_ep{ep:06d}.parquet'
            expected = loader.get_episode_length(ep)
            if target.exists():
                validate_episode(target, ep, expected)
                remaining_frames -= expected
                continue
            random.seed(a.seed + ep)
            np.random.seed(a.seed + ep)
            torch.manual_seed(a.seed + ep)
            started = time.perf_counter()
            data = loader[ep]
            assert len(data) == expected, (ep, len(data), expected)
            def prepare(start):
                items = []
                for frame in range(start, min(start + a.batch_size, expected)):
                    step = extract_step_data(data, frame, config, EmbodimentTag.GENERAL_EMBODIMENT, True)
                    items.append(policy.processor([{'type': 'episode_step', 'content': step}]))
                return policy.collate_fn(items)
            future = pool.submit(prepare, 0)
            aggregator = CausalEntropy(horizon)
            rows = []
            for start in range(0, expected, a.batch_size):
                batch = future.result()
                if start + a.batch_size < expected:
                    future = pool.submit(prepare, start + a.batch_size)
                if not verified:
                    verify_sampling_path(policy, batch)
                samples = sample_chunks(policy, batch, a.n_samples)[..., :real_dim]
                if not np.isfinite(samples).all():
                    raise ValueError(f'Non-finite samples in episode {ep}, frame {start}')
                if not verified:
                    assert float(samples.std(axis=1).mean()) > 0, 'Identical stochastic samples'
                    verified = True
                for offset, sample in enumerate(samples):
                    frame = start + offset
                    rows.append(dict(episode=ep, frame=frame, **aggregator.append(frame, sample)))
            result = pd.DataFrame(rows)
            result['entropy_z'] = (result.kde_bw1_agg - result.kde_bw1_agg.mean()) / max(float(result.kde_bw1_agg.std(ddof=0)), 1e-12)
            tmp = target.with_suffix('.parquet.partial')
            result.to_parquet(tmp, index=False)
            tmp.replace(target)
            elapsed = time.perf_counter() - started
            measured_frames += expected
            measured_seconds += elapsed
            remaining_frames -= expected
            rate = measured_frames / measured_seconds
            print('ENTROPY_PROGRESS ' + json.dumps(dict(episode=ep, frames=expected,
                  seconds=elapsed, frames_per_second=rate, remaining_frames=remaining_frames,
                  eta_hours=remaining_frames / rate / 3600,
                  entropy_std=float(result.kde_bw1_agg.std(ddof=0)))), flush=True)
    print('ENTROPY_COMPLETE ' + json.dumps(dict(shard=a.shard_index, episodes=len(episodes))), flush=True)

if __name__ == '__main__':
    main()
