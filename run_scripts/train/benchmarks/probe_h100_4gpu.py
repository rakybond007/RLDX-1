"""Check CUDA contexts and NCCL before starting a four-GPU H100 training job."""

import os
import torch
import torch.distributed as dist


rank = int(os.environ["LOCAL_RANK"])
print(f"CUDA_PROBE_START rank={rank} visible={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
torch.cuda.set_device(rank)
device = torch.device("cuda", rank)
value = torch.tensor([float(rank)], device=device)
free, total = torch.cuda.mem_get_info(rank)
print(f"CUDA_PROBE_CONTEXT rank={rank} device={torch.cuda.get_device_name(rank)} free={free} total={total}", flush=True)
dist.init_process_group("nccl")
dist.all_reduce(value)
torch.cuda.synchronize(device)
assert value.item() == 6.0
print(f"CUDA_PROBE_OK rank={rank} sum={value.item()}", flush=True)
dist.destroy_process_group()
