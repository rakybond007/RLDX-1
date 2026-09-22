"""Opt-in throughput/resume validation using the unchanged production LR schedule.

Run through a training shell script with TRAIN_ENTRYPOINT pointing here and
BENCHMARK_STEPS set. Use a separate output directory containing a symlink to
a complete checkpoint to validate resume without modifying the production run.
"""
import json
import math
import os
from pathlib import Path
import runpy
import time

import torch
from transformers import TrainerCallback

from rldx.experiment.trainer import RLDXTrainer


class ThroughputProbe(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_step = state.global_step
        self.steps = int(os.environ["BENCHMARK_STEPS"])
        if self.steps <= 20:
            raise ValueError("BENCHMARK_STEPS must exceed the 20-step warmup")
        self.last = time.perf_counter()
        self.intervals = []
        self.norm_ab = os.environ.get("BENCHMARK_RMSNORM_AB") == "1"
        self.identity_ab = os.environ.get("BENCHMARK_IMAGE_IDENTITY_AB") == "1"
        if self.identity_ab and (self.steps != 140 or os.environ.get("RLDX_IMAGE_IDENTITY_FASTPATH") != "1"):
            raise ValueError("Image identity A/B requires 140 updates and the fast path enabled")
        if self.norm_ab and (self.steps != 140 or os.environ.get("RLDX_COMPILE_RMSNORM") != "1"):
            raise ValueError("RMSNorm A/B requires BENCHMARK_STEPS=140 and RLDX_COMPILE_RMSNORM=1")
        torch.cuda.reset_peak_memory_stats()
        if state.is_world_process_zero:
            print("BENCHMARK_START " + json.dumps({
                "resume_step": self.start_step,
                "test_updates": self.steps,
                "schedule_max_steps": args.max_steps,
                "batch_per_device": args.per_device_train_batch_size,
                "accumulation": args.gradient_accumulation_steps,
                "workers_per_rank": args.dataloader_num_workers,
            }), flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.intervals.append(now - self.last)
        self.last = now
        completed = state.global_step - self.start_step
        if self.identity_ab and completed in (60, 100):
            from rldx.model.modules.backbone.layer_wrapper import LayerWrapper
            for module in kwargs["model"].modules():
                if isinstance(module, LayerWrapper):
                    module.image_identity_fastpath = completed == 100
        if self.norm_ab and completed in (60, 100):
            from rldx.utils.training_kernels import set_compiled_text_norms

            set_compiled_text_norms(kwargs["model"], enabled=completed == 100)
        if state.is_world_process_zero and completed % 10 == 0:
            window = self.intervals[-10:]
            print("BENCHMARK_PROGRESS " + json.dumps({
                "step": state.global_step, "updates": completed,
                "seconds_per_step_last10": sum(window) / len(window),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            }), flush=True)
        if completed >= self.steps:
            measured = self.intervals[20:]
            if state.is_world_process_zero:
                if self.norm_ab or self.identity_ab:
                    ranges = {"compiled_first": (25, 60), "eager": (65, 100),
                              "compiled_second": (105, 140)}
                    label = "BENCHMARK_IMAGE_IDENTITY_AB" if self.identity_ab else "BENCHMARK_RMSNORM_AB"
                    print(label + " " + json.dumps({
                        name: sum(self.intervals[a:b]) / (b-a)
                        for name, (a, b) in ranges.items()
                    }), flush=True)
                print("BENCHMARK_RESULT " + json.dumps({
                    "start_step": self.start_step, "end_step": state.global_step,
                    "warmup_steps": 20, "measured_steps": len(measured),
                    "mean_seconds_per_step": sum(measured) / len(measured),
                    "p90_seconds_per_step": sorted(measured)[int(0.9 * (len(measured) - 1))],
                    "max_seconds_per_step": max(measured),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                }), flush=True)
            control.should_training_stop = True
            control.should_save = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        for key in ("loss", "grad_norm"):
            if logs and key in logs and not math.isfinite(float(logs[key])):
                raise RuntimeError(f"Non-finite {key}: {logs[key]}")


class OperatorProfile(TrainerCallback):
    """Profile three warm steps; finish before throughput's warmup ends."""

    def on_train_begin(self, args, state, control, **kwargs):
        self.prof = None
        if not state.is_world_process_zero:
            return

        def report(prof):
            folder = Path(args.output_dir) / "profiling"
            folder.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(folder / "rank0.json"))
            print("BENCHMARK_OPERATOR_PROFILE\n" + prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=35,
            ), flush=True)

        self.prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                skip_first=10, wait=1, warmup=1, active=3, repeat=1,
            ),
            on_trace_ready=report,
        )
        self.prof.start()

    def on_step_end(self, args, state, control, **kwargs):
        if self.prof is not None:
            self.prof.step()

    def on_train_end(self, args, state, control, **kwargs):
        if self.prof is not None:
            self.prof.stop()


if __name__ == "__main__":
    original_init = RLDXTrainer.__init__

    def with_probe(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.add_callback(ThroughputProbe())
        if os.environ.get("BENCHMARK_PROFILE") == "1":
            self.add_callback(OperatorProfile())

    RLDXTrainer.__init__ = with_probe
    root = Path(__file__).resolve().parents[2]
    runpy.run_path(str(root / "rldx/experiment/launch_train.py"), run_name="__main__")
