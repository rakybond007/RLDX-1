"""GPU forward/backward equivalence and timing for the compiled text RMSNorm.

Run from the repository root with PYTHONPATH=. .venv/bin/python
 tests/check_compiled_rmsnorm.py. Covers frozen and trainable norm weights.
"""
import copy
import json
import time

import torch

from rldx.model.modules.backbone.modeling_qwen3_vl import Qwen3VLTextRMSNorm
torch.cuda.set_device(0)
torch.manual_seed(13)
results = []
for train_weight in [False, True]:
    eager = Qwen3VLTextRMSNorm(4096).cuda().bfloat16()
    eager.weight.requires_grad_(train_weight)
    compiled = copy.deepcopy(eager)
    compiled.forward = torch.compile(compiled.forward, fullgraph=True, dynamic=True)
    x = torch.randn(32, 320, 4096, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    grad = torch.randn_like(x)
    a = eager(x)
    a.backward(grad)
    b = compiled(y)
    b.backward(grad)
    checks = {'output': (a, b), 'input_grad': (x.grad, y.grad)}
    if train_weight:
        checks['weight_grad'] = (eager.weight.grad, compiled.weight.grad)
    errors = {}
    for (name, (v, w)) in checks.items():
        err = ((v.float() - w.float()).norm() / v.float().norm().clamp_min(1e-08)).item()
        assert err < 0.015, (name, err)
        errors[name] = err
    del a, b, checks
    timings = {}
    for (name, model, inp) in [('eager', eager, x), ('compiled', compiled, y)]:

        def step():
            model.zero_grad(set_to_none=True)
            inp.grad = None
            model(inp).backward(grad)
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(15):
            step()
        torch.cuda.synchronize()
        timings[name] = (time.perf_counter() - start) / 15
    results.append({'train_weight': train_weight, 'seconds': timings, 'relative_l2_errors': errors})
print('RMSNORM_BENCHMARK ' + json.dumps(results), flush=True)
