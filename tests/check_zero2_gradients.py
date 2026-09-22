"""Two-GPU regression probe for the baseline's ZeRO-2/BF16 gradient path.

Run with .venv/bin/torchrun --standalone --nproc_per_node=2
tests/check_zero2_gradients.py. Uses distinct rank-local inputs and compares
every parameter's gradient with an explicitly averaged reference for 3 updates.
This checks the installed stack, not full RLDX model gradient equivalence.
"""
import copy
import json
import os
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.utils import safe_get_full_grad


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, "This probe requires two GPUs"
    torch.manual_seed(123)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 8),
    ).cuda().to(torch.bfloat16)
    reference = copy.deepcopy(model)
    cfg = json.loads((Path(__file__).resolve().parents[1] /
                      "rldx/configs/deepspeed/zero2_config.json").read_text())
    cfg.update(train_batch_size=8, train_micro_batch_size_per_gpu=4,
               gradient_accumulation_steps=1, gradient_clipping=1.0)
    cfg["fp16"]["enabled"] = False
    cfg["bf16"]["enabled"] = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    engine, _, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config=cfg,
    )
    maximum = 0.0
    for step in range(3):
        reference.load_state_dict(engine.module.state_dict())
        reference.zero_grad(set_to_none=True)
        torch.manual_seed(1000 + rank + step * world)
        x = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
        target = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
        ref_loss = (reference(x).float() - target.float()).square().mean()
        ref_loss.backward()
        loss = (engine(x).float() - target.float()).square().mean()
        engine.backward(loss)
        errors = []
        for p, q in zip(engine.module.parameters(), reference.parameters()):
            # Match production's BF16 communication before converting to FP32.
            expected = q.grad.detach().clone()
            dist.all_reduce(expected)
            expected = expected.float() / world
            actual = safe_get_full_grad(p).float()
            error = ((actual - expected).norm() /
                     expected.norm().clamp_min(1e-8)).item()
            assert torch.isfinite(actual).all(), "Non-finite reduced gradient"
            assert error < 0.015, f"Gradient mismatch: relative L2={error}"
            errors.append(error)
        maximum = max(maximum, *errors)
        engine.step()
    if rank == 0:
        print("ZERO2_GRADIENT_CHECK_PASSED " + json.dumps({
            "deepspeed": deepspeed.__version__, "world_size": world,
            "accumulation": 1, "bf16": True, "steps": 3,
            "maximum_relative_l2_error": maximum,
        }), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
