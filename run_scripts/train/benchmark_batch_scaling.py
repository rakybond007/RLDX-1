"""Disposable 2-GPU RoboCasa batch scaling probe; never save its updates.

Keep the production loader at 32 samples/rank and select complete examples for
8/16/32-sample compute probes. Padding length and loader work stay fixed. This
measures compute/optimizer scaling, not the smaller batch's loader throughput.
Do not promote this run's state into the global-64 baseline.
"""
import json
from pathlib import Path
import runpy
import time

import torch
from transformers import TrainerCallback
from rldx.experiment.trainer import RLDXTrainer


def select_examples(batch, size):
    values = batch['inputs']
    count = values['input_ids'].shape[0]
    if not (size <= count and count == 32):
        raise ValueError('This probe expects the unchanged 32-example/rank loader')
    if not bool((values['num_frames'] == 1).all()):
        raise ValueError('Image-only probe')
    views = values['num_views']
    images = int(views[:size].sum())
    grid = values['image_grid_thw']
    assert grid.shape[0] == int(views.sum())
    patches = int(grid[:images].prod(-1).sum())
    assert values['pixel_values'].shape[0] == int(grid.prod(-1).sum())
    selected = {}
    for key, value in values.items():
        if key == 'pixel_values':
            selected[key] = value[:patches]
        elif key == 'image_grid_thw':
            selected[key] = value[:images]
        else:
            if not isinstance(value, torch.Tensor) or value.shape[0] != count:
                raise ValueError(f'Unexpected batch field: {key}')
            selected[key] = value[:size]
    return {'inputs': selected}


class ScalingProbe(TrainerCallback):
    sizes = (8, 16, 32)

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = state.global_step
        self.last = time.perf_counter()
        self.intervals = []
        print('BATCH_SCALING_DISPOSABLE_NO_CHECKPOINTS', flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.intervals.append(now - self.last)
        self.last = now
        done = state.global_step - self.start
        if done % 40 == 0 and state.is_world_process_zero:
            seconds = sum(self.intervals[-30:]) / 30
            size = self.sizes[done // 40 - 1]
            print('BATCH_SCALING_RESULT ' + json.dumps({
                'batch_per_rank': size, 'global_batch': size * 2,
                'seconds_per_step': seconds, 'samples_per_second': size * 2 / seconds,
                'measured_steps': 30, 'warmup_per_phase': 10,
            }), flush=True)
        control.should_save = False
        if done >= 120:
            control.should_training_stop = True
        return control


if __name__ == '__main__':
    original_init = RLDXTrainer.__init__
    original_step = RLDXTrainer.training_step

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._scaling_probe = ScalingProbe()
        self.add_callback(self._scaling_probe)
        # This experiment changes batch size intentionally and must never be resumed.
        self.args.save_strategy = 'no'

    def step(self, model, inputs, *args, **kwargs):
        phase = (self.state.global_step - self._scaling_probe.start) // 40
        return original_step(self, model, select_examples(inputs, ScalingProbe.sizes[phase]),
                             *args, **kwargs)

    RLDXTrainer.__init__ = initialize
    RLDXTrainer.training_step = step
    RLDXTrainer.save_model = lambda *args, **kwargs: print(
        'BATCH_SCALING: discarded model; no baseline checkpoint written', flush=True)
    root = Path(__file__).resolve().parents[2]
    runpy.run_path(str(root / 'rldx/experiment/launch_train.py'), run_name='__main__')
