"""Opt-in training kernels that retain model parameters and checkpoint keys."""
import json
import math
import time
from types import MethodType

import torch
from transformers import TrainerCallback


class ResumeSaveInterval(TrainerCallback):
    """Apply the requested save cadence after Trainer reloads checkpoint state."""

    def on_train_begin(self, args, state, control, **kwargs):
        state.save_steps = (math.ceil(args.save_steps * state.max_steps)
                            if args.save_steps < 1 else int(args.save_steps))
        if state.is_world_process_zero:
            print(f"Checkpoint save interval after resume: {state.save_steps}", flush=True)


class TrainingThroughput(TrainerCallback):
    """Report actual resumed update throughput after twenty warmup updates."""

    def on_train_begin(self, args, state, control, **kwargs):
        self.warmup_end = state.global_step + 20
        self.last_step = self.warmup_end
        self.last_time = None

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or state.global_step < self.warmup_end:
            return
        if self.last_time is None:
            torch.cuda.synchronize()
            self.last_time = time.perf_counter()
            return
        updates = state.global_step - self.last_step
        if updates >= 50:
            torch.cuda.synchronize()
            now = time.perf_counter()
            print("TRAINING_THROUGHPUT " + json.dumps({
                "global_step": state.global_step,
                "measured_updates": updates,
                "seconds_per_step": (now - self.last_time) / updates,
            }), flush=True)
            self.last_step = state.global_step
            self.last_time = now


def set_compiled_text_norms(model, enabled=True):
    from rldx.model.modules.backbone.modeling_qwen3_vl import Qwen3VLTextRMSNorm

    count = 0
    for module in model.modules():
        # Target the large residual-stream norms measured in the training probe.
        if not isinstance(module, Qwen3VLTextRMSNorm) or module.weight.numel() != 4096:
            continue
        if not hasattr(module, "_rldx_eager_forward"):
            module._rldx_eager_forward = module.forward
            module._rldx_compiled_forward = torch.compile(
                module.forward, fullgraph=True, dynamic=True,
            )
        module.forward = (module._rldx_compiled_forward if enabled
                          else module._rldx_eager_forward)
        count += 1
    return count


def _cpu_rope_index(self, input_ids=None, image_grid_thw=None,
                    video_grid_thw=None, attention_mask=None):
    values = (input_ids, image_grid_thw, video_grid_thw, attention_mask)
    if input_ids is None or input_ids.device.type != "cuda":
        return self._rldx_original_rope_index(*values)
    device = input_ids.device
    # Position indices have no gradients. The original routine already builds
    # positions on CPU, but repeatedly synchronizes individual CUDA scalars.
    cpu_values = tuple(x.cpu() if x is not None else None for x in values)
    positions, deltas = self._rldx_original_rope_index(*cpu_values)
    return positions.to(device), deltas.to(device)


def enable_cpu_rope_metadata(model):
    from rldx.model.modules.backbone.modeling_qwen3_vl import Qwen3VLModel

    count = 0
    for module in model.modules():
        if not isinstance(module, Qwen3VLModel):
            continue
        if not hasattr(module, "_rldx_original_rope_index"):
            module._rldx_original_rope_index = module.get_rope_index
            module.get_rope_index = MethodType(_cpu_rope_index, module)
        count += 1
    return count
