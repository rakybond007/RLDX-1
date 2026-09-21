# RoboCasa Kitchen image-input training on two GPUs

Run all commands from the RLDX-1 repository root. The environment is isolated
in `.venv`; model weights, caches and run outputs are git-ignored.

```bash
bash run_scripts/train/setup_uv.sh
mkdir -p out
```

The recipe uses `RLWRLD/RLDX-1-PT-IMG`, `video_length=1`, all three camera
views, a 16-step action horizon, and the repository's default fine-tuning
parameter selection. It uses the existing RoboCasa Kitchen 300-demo LeRobot
dataset (not RoboCasa365). Override `DATA_DIR` on other machines.

Default effective batch: **2 GPUs × 32 examples × 1 accumulation = 64**.
`GRAD_ACCUM` can be increased if needed; the microbatch is then reduced so the
effective batch stays 64. The batch resolver includes accumulation and rejects
non-divisible values. Legacy `TrainingConfig.batch_size` still explicitly
sets the per-device microbatch.

Inside an existing two-GPU `srun` allocation:

```bash
MAX_STEPS=10 SAVE_STEPS=10 RUN_NAME=smoke_img_gb64 \
  bash run_scripts/train/benchmarks/finetune_rldx1_robocasa_image_2gpu.sh \
  > out/smoke_img_gb64.log 2>&1
```

Use `BASE_MODEL_PATH=$PWD/models/RLDX-1-PT-IMG` if the checkpoint was already
downloaded there. Otherwise the launcher downloads from Hugging Face.

For a full 60,000-step run (submit only when ready):

```bash
sbatch run_scripts/train/benchmarks/finetune_rldx1_robocasa_image_2gpu.sh
```

The default Slurm partition/accounting settings follow the local GR00T
recipes. Override resource settings using `sbatch` options on another cluster.
`OUTPUT_DIR` is the output parent; the launcher adds `RUN_NAME` beneath it.
Other overrides: `MAX_STEPS`, `SAVE_STEPS`, `NUM_WORKERS`, `GRAD_ACCUM`.
The standard random-access dataset mode avoids preloading large episode shards
for a short smoke run. WandB is disabled by default.

CPU regression check:

```bash
python3 tests/test_effective_batch.py
```

## Verified smoke result (2026-09-21 KST)

- Existing Slurm allocation 202026, worker-node107, 2 × A100-SXM4-80GB.
- Python 3.10.12, PyTorch 2.7.0+cu126, FlashAttention 2.7.4.post1;
  GPU computation and torchcodec imports passed.
- Base checkpoint revision: `d67fc642a7e768049ba41ff8cf0dfe0ff17f556b`.
- 7,200 episodes / 1,965,457 dataset steps; three current-frame camera views.
- 10 optimizer steps, batch 32 per GPU, accumulation 1, global batch 64.
- Mean loss 0.27646484375; step-10 gradient norm 0.46398094296455383.
- Observed GPU memory: 68,603 / 69,243 MiB (a sampled reading, not peak).
- Steps completed in ~55 seconds; Trainer runtime including checkpoint save
  was 166.98 seconds. Initial model construction and final model export are
  additional. Random-access data loading caused intermittent stalls.
- `checkpoint-10` contains model shards, optimizer state, trainer state,
  processor and experiment configuration. All 722 indexed model tensors fit
  their shard bounds; the trainable `backbone.cog_emb` changed from the base.
- Training and final export completed with shell exit code 0. This verifies
  the training path, not downstream policy success or convergence.
- Log: `out/smoke_img_gb64.log`; artifacts: `outputs/smoke_img_gb64/`.
