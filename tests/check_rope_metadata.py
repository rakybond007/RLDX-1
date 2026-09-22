"""GPU exact-equality and timing check for RoboCasa's CPU RoPE metadata path."""
import json
import time
from types import SimpleNamespace
import torch
from rldx.model.modules.backbone.modeling_qwen3_vl import Qwen3VLModel
from rldx.utils.training_kernels import enable_cpu_rope_metadata
model = Qwen3VLModel.__new__(Qwen3VLModel)
torch.nn.Module.__init__(model)
model.config = SimpleNamespace(
    vision_config=SimpleNamespace(spatial_merge_size=2),
    image_token_id=101,
    video_token_id=102,
    vision_start_token_id=100,
)
rows = []
for i in range(32):
    row = [5] * (10 + i % 13)
    for _ in range(3):
        row += [100] + [101] * 64 + [103]
    row += [248068] * 64
    rows.append(row)
length = max(map(len, rows))
ids = torch.zeros(32, length, dtype=torch.long)
mask = torch.zeros_like(ids)
for (i, row) in enumerate(rows):
    ids[i, -len(row):] = torch.tensor(row)
    mask[i, -len(row):] = 1
grid = torch.tensor([[1, 16, 16]] * 96)
(ids, mask, grid) = (ids.cuda(), mask.cuda(), grid.cuda())

def original():
    return Qwen3VLModel.get_rope_index(model, ids, grid, attention_mask=mask)
assert enable_cpu_rope_metadata(model) == 1

def cpu_metadata():
    return model.get_rope_index(ids, grid, attention_mask=mask)
a = original()
b = cpu_metadata()
assert all((torch.equal(v, w) for (v, w) in zip(a, b)))
r = {}
for (name, fn) in [('original', original), ('cpu_metadata', cpu_metadata)]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    r[name] = (time.perf_counter() - start) / 10
print('ROPE_BENCHMARK ' + json.dumps({'seconds': r, 'exact_match': True}), flush=True)
