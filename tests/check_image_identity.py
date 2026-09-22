"""Verify image-only token compression is an exact forward/backward identity."""
import json
import time
import torch
from rldx.model.modules.backbone.layer_wrapper import LayerWrapper

class Probe(torch.nn.Module):
    def forward(self, x, **kwargs):
        return x.square()

def check(frames=1, motion=False, timing=False):
    batch, length, dim = (32, 320, 4096) if timing else (3, 40, 8)
    ids = torch.ones(batch, length, dtype=torch.long, device='cuda')
    for b in range(batch):
        ids[b, torch.arange(3 * frames, device='cuda') * 3 + 2 + b % 2] = 151652
    metadata = dict(image_wise_encoding=torch.ones(batch, device='cuda'),
                    num_views=torch.full((batch,), 3, device='cuda'),
                    attention_mask=torch.ones_like(ids), attention_mask_2=torch.ones_like(ids),
                    position_ids=torch.arange(length, device='cuda').expand(3, batch, -1),
                    position_embeddings=(torch.randn(batch, length, 8, device='cuda'),
                                         torch.randn(batch, length, 8, device='cuda')),
                    cache_position=torch.arange(length, device='cuda'),
                    motion_drop_info={'start': 0, 'count': 1} if motion else None)
    wrapper = LayerWrapper(Probe(), 4, motion_token=1).cuda()
    seed = torch.randn(batch, length, dim, device='cuda', dtype=torch.bfloat16)
    outputs = []
    times = {}
    for enabled in (False, True):
        wrapper.image_identity_fastpath = enabled
        x = seed.clone().requires_grad_()
        result, kw = wrapper(x, ids, **metadata)
        result.float().sum().backward()
        outputs.append((result.detach(), x.grad, kw))
        if timing:
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                wrapper(x, ids, **metadata)[0].float().sum().backward()
                x.grad = None
            torch.cuda.synchronize()
            times[str(enabled)] = (time.perf_counter() - start) / 10
    a, b = outputs
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    for key in a[2]:
        left, right = a[2][key], b[2][key]
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right), key
        elif isinstance(left, tuple):
            assert all(torch.equal(v, w) for v, w in zip(left, right)), key
        else:
            assert left == right, key
    return times

check()
check(frames=2)
check(motion=True)
print('IMAGE_IDENTITY_CHECK ' + json.dumps({'exact_forward_backward': True,
      'seconds': check(timing=True)}), flush=True)
