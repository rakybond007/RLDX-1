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
The RoboCasa launcher requires `GRAD_ACCUM=1` because pinned DeepSpeed 0.17.6
has a reported ZeRO-2 accumulation regression (issue #7718). Increasing it
requires a separately validated DeepSpeed environment. The general batch resolver includes accumulation and rejects
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
sbatch --export=ALL,MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/checkpoints/rldx1_image_baselines \
  run_scripts/train/benchmarks/finetune_rldx1_robocasa_image_2gpu.sh
```

The default Slurm partition is `sjw_alinlab_premium`, with local GR00T
accounting settings. CPU resources are assigned automatically by the cluster. Override resource settings using `sbatch` options on another cluster.
`OUTPUT_DIR` (falling back to `MODEL_OUTPUT_DIR`) is the output parent; the launcher adds `RUN_NAME` beneath it.
Other overrides: `MAX_STEPS`, `SAVE_STEPS`, `NUM_WORKERS`, `GRAD_ACCUM`.
Both baselines use the original `sharded` dataset mode, with background
prefetch and GPU-local workers (RoboCasa 12, LIBERO 8). WandB is disabled by default.

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

## Premium baseline submissions (2026-09-22 KST)

| Dataset | Job | GPUs | Batch per GPU | Global batch | Baseline steps |
|---|---|---|---|---|---|
| RoboCasa Kitchen | 203169 | 2 | 32 | 64 | 60,000 |
| LIBERO (40 tasks) | 203170 | 2 | 16 | 32 | 60,000 |

Both use PT-IMG, one frame per camera, action horizon 16, accumulation 1,
and the `sjw_alinlab_premium` partition (48-hour time limit). They prefer the
existing local `models/RLDX-1-PT-IMG` download. The output parent is
`/rlwrld-unified-checkpoints/hojin2/checkpoints/rldx1_image_baselines`; run
names are `rldx1_img_robocasa_gb64_60k_baseline` and
`rldx1_img_libero_gb32_60k_baseline`. RoboCasa checkpoints are saved every 2,000 steps; LIBERO retains its
1,000-step cadence. Both retain two checkpoints. The launcher resumes if the same run already has a valid
checkpoint.

LIBERO uses `finetune_rldx1_libero_image_2gpu.sh` and the existing
`libero_gr00t_delta` dataset (1,693 episodes, 273,465 frames, 40 tasks).
The batch script starts the full baseline directly. LIBERO smoke is performed
separately in the user's existing interactive GPU allocation, with an isolated
output directory, before submitting the full baseline.

```bash
sbatch --export=ALL,MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/checkpoints/rldx1_image_baselines \
  run_scripts/train/benchmarks/finetune_rldx1_libero_image_2gpu.sh
```

## Verified LIBERO smoke (2026-09-22 KST)

Ran manually in tmux `1:1`, allocation 202544 on worker-node113, using two
A100-SXM4-80GB GPUs. The previous gated job 202540 was cancelled; after the
manual smoke passed, full training was submitted as job 202568. RoboCasa
job 202539 was left unchanged.

- PT-IMG, current frame from both cameras, action chunk 16.
- Batch 16 per GPU, accumulation 1, effective global batch 32.
- 1,693 episodes / 248,070 valid training samples after horizon filtering.
- 10 optimizer steps passed; loss 0.341943359375, gradient norm 1.0940557718.
- ~18 seconds for the steps, 127.57 seconds including checkpoint saving
  (initialization and final export are additional).
- GPU memory sampled just after the last step: 49,807 / 50,011 MiB.
- Checkpoint and final export completed, shell exit code 0. All 722 indexed
  tensor extents validated; trainable `backbone.cog_emb` changed from PT-IMG.
- Log: `out/smoke_libero_img_gb32_202544.log`.
- Artifacts: `outputs/smoke_libero_img_gb32_202544/`.

This verifies loading, forward/backward, optimizer updates and saving; it
does not measure policy success or long-run convergence.

## Loader correction and checkpoint resume (2026-09-22)

The original submitted runs incorrectly retained the smoke setup's `standard`
map-style loader with four workers per rank. Random sampling repeatedly loaded
and decoded whole episodes while the single-episode cache rarely hit. RoboCasa
ran at ~4.63 seconds/step overall, ~5.16 seconds/step over the last 200 steps.

Corrected settings match the upstream training recipes:

| Setting | RoboCasa | LIBERO |
|---|---|---|
| Dataset mode | sharded | sharded |
| Workers per GPU | 12 | 8 |
| Global batch | 64 | 32 |
| Gradient accumulation | 1 | 1 |
| Shard size | 1024 | 1024 |
| Episode sampling rate | 0.1 | 0.1 |
| RTC training / inference | 0 / none | 0 / none |

TorchCodec now defaults to one FFmpeg thread per decoder, preventing every
worker from automatically creating a large decoder thread pool. An explicit
`video_backend_kwargs["num_ffmpeg_threads"]` still overrides this. The regression
test `tests/test_video_decode_threads.py` verifies identical pixels/timestamps
versus automatic threading and indexed decoding with an explicit override.
OMP/MKL/OpenBLAS threads are also bounded in the launch scripts.

Old jobs 202539 and 202568 were stopped after verifying complete model,
processor, trainer, and both optimizer partitions at checkpoints 3000 and
15000 respectively. Jobs 203169 and 203170 retain the same output directories
and use the trainer's checkpoint resume, preserving optimizer/scheduler state.
Updates after those checkpoints must be repeated. The changed sampler order
is not bitwise reproducible with the old loader.

Learning rate 1e-4, warmup ratio 0.05, weight decay 1e-5, top four LLM layers
and action-head fine-tuning remain the repository defaults. The explicit image
checkpoint, one-frame input, 16-step action horizon, and user-requested global
batches are unchanged. Checkpoint retention remains two and WandB remains
disabled, as in the initial submitted scripts.


### RoboCasa sustained resume validation

Allocation 203166, worker-node1, 2 x A100 80GB PCIe (PHB connection; no
NVLink). The previous full run used a different node with A100 SXM GPUs,
so the before/after comparison is not hardware-controlled.

`TRAIN_ENTRYPOINT=run_scripts/train/benchmark_training.py BENCHMARK_STEPS=450`
kept `MAX_STEPS=60000`, resumed checkpoint 3000 and stopped at 3450. This
preserves the full-run LR schedule rather than substituting a short schedule.

- 20 warmup updates excluded; 430 timed updates averaged 2.323172 s/step.
- p90 2.409831 s; maximum 3.163173 s; rank-0 peak allocation 56.289 GiB.
- All 24 workers consumed two shards; subsequent shard-cache waits were
  0.00 s at the log's 0.01 s resolution.
- Final logged loss 0.0876, gradient norm 0.213529; finite logged metrics.
- Normal exit 0; checkpoint 3450 contains 722 indexed model tensors,
  both optimizer partitions at step 3450, and scheduler epoch 3450.
- Checkpoint saving added approximately 144 seconds; final model export
  is additional. The long-run extrapolation is provisional: approximately
  39-40 hours from step 3000 including 1000-step checkpoint saves.
- Log: `out/validate_robocasa_sharded_resume_203166.log`.

LIBERO job 203170 was released at the user's request, resuming checkpoint
15000. Its proposed additional interactive benchmark was skipped. Subsequent
profiling and correctness probes apply only to RoboCasa's debug allocation.

### DeepSpeed correctness check

The environment is pinned to DeepSpeed 0.17.6. The RoboCasa launcher rejects
accumulation greater than one due to [issue #7718](https://github.com/deepspeedai/DeepSpeed/issues/7718).
The separate [issue #8224](https://github.com/deepspeedai/DeepSpeed/issues/8224)
reports a later regression; its [fix #8245](https://github.com/deepspeedai/DeepSpeed/pull/8245)
addresses scaling shared-loss branches with gradient accumulation.

`tests/check_zero2_gradients.py` exercised two ranks, BF16, ZeRO-2 with the
production communication settings, accumulation 1, distinct rank-local inputs,
and three updates. Every parameter's reduced gradient matched the explicit
BF16 all-reduce reference (maximum relative L2 difference 0.0). This is a
small-model regression probe of the installed stack, not proof of complete
RLDX model gradient equivalence. Log: `out/check_zero2_gradients_203166.log`.


### Operator profile and targeted optimization probes

A separate 60-update resume from checkpoint 3450 retained the 60k schedule.
The profiler recorded three steps after warmup; subsequent 40 unprofiled
updates averaged 2.186406 s/step. Profile trace (git-ignored):
`outputs/profile_robocasa_sharded_203166/profiling/rank0.json`.

- DataLoader next-batch CPU time: 0.132, 0.553, 0.344 ms (mean 0.343 ms).
- NCCL GPU kernels: 1610.708 ms over three steps, about 536.903 ms/step.
  These kernel durations can overlap compute and are not additive wall time.
- Same-GPU collective microbenchmark: forcing NCCL Simple reduced the large
  optimizer all-gather from 178.2 to 148.0 ms, but the expected overall gain
  is small. It is enabled for RoboCasa after the user requested all verified
  optimizations that preserve the model.
- Moving RoPE metadata computation to CPU produced identical integer positions
  and saved ~12.7 ms/call. It is enabled for RoboCasa, with the actual
  wrapper checked by `tests/check_rope_metadata.py`.
- Compiling the residual-stream RMSNorm forward/backward reduced representative
  32x320x4096 BF16 latency from 3.52-3.74 ms to 0.47 ms. Relative L2 error was
  <=1.49e-5 for output/input gradients and 0.00311 for trainable weight
  gradients. `tests/check_compiled_rmsnorm.py` preserves this check.
- Full-training A/B validation used the same process, alternating compiled,
  eager, then compiled norms over 140 resumed updates (3510 to 3650).
  The three measured intervals averaged 2.054321, 2.178012, and 2.062350
  seconds/step respectively: approximately 5.5% faster with compiled norms.
  Each interval excluded warmup/transition updates. `RLDX_COMPILE_RMSNORM=1` is opt-in and keeps parameter
  names/shapes and checkpoint keys unchanged. It targets only 4096-wide text
  residual-stream norms, not the model architecture or training objective.


The optimized RoboCasa recipe additionally uses `NCCL_PROTO=Simple`,
`RLDX_CPU_ROPE=1`, and `RLDX_COMPILE_RMSNORM=1`. These change execution,
not camera selection/resolution, action horizon, parameter selection, model
weights layout, global batch, LR schedule, or loss. Floating-point kernel
fusion is checked within BF16 tolerance, not claimed bitwise identical.

RoboCasa saves every 2000 steps to reduce observed 2-minute save stalls.
`RLDX_REFRESH_SAVE_INTERVAL=1` reapplies the requested interval after Trainer
loads checkpoint state (which otherwise retains the old 1000-step interval).
`TRAINING_THROUGHPUT` records real 50-update wall times after 20 warmup updates,
avoiding misleading progress-bar rates immediately after checkpoint resume.

### Image-only overhead and batch scaling probes

The layer-4 video token compression path also executes for current-frame input.
With one image per view and no motion tokens to drop, it reconstructs exactly
its input, including masks and RoPE tensors. `RLDX_IMAGE_IDENTITY_FASTPATH=1`
checks this condition and skips reconstruction; video/motion cases retain the
original path. It is off unless explicitly enabled.
`tests/check_image_identity.py` checks exact outputs, input gradients, and all
metadata for image-only, video, and motion-drop cases. On allocation 203166,
the 32x320x4096 forward/backward probe measured 57.473 ms original versus
1.159 ms with the identity path. This isolated saving is not a measured
end-to-end training improvement. A 140-update training A/B is recorded in
`out/ab_image_identity_robocasa_203166.log` when complete.

`run_scripts/train/benchmark_batch_scaling.py` measures 8/16/32 examples per
rank, 40 updates per phase with the first 10 excluded. It selects complete
examples and their corresponding image patches from the original batch of 32;
padding length and loader work stay fixed. Its updates are disposable and
model saving is disabled. Never promote its state into the global-64 baseline.
Use a separate output directory and `TRAIN_ENTRYPOINT` to invoke it.

Allocation 203166 has A100 PCIe GPUs without NVLink, whereas the resumed
LIBERO allocation has A100 SXM GPUs. Cross-run timing is therefore not a
controlled comparison. The user's GR00T 16-hour/60k reference corresponds to
0.96 seconds/update including checkpoint overhead, but is not a required
RLDX timing target. Likewise, matching LIBERO step time is not the goal:
identify avoidable overhead and justify the remaining cost with measurements.
The GR00T N1.5 reference
script uses a frozen LLM/vision tower, while RLDX trains its top four LLM
layers and cognition embeddings. Preserve that parameter selection during
optimization; it is a workload difference, not evidence that current speed
is optimal.

The image identity A/B completed 140 updates (3650 to 3790). Excluding
transition/warmup updates, ON/OFF/ON measured 2.042185 / 2.124848 / 2.085250
seconds/update. The average ON interval is 2.063718 seconds (31.01 samples/s),
about 2.88% faster than OFF. Final loss was 0.0795 and grad norm 0.291827.
The RoboCasa launcher now enables the identity path by default; the underlying
module remains opt-in for all other launchers. This is a modest improvement,
not an explanation of the full performance gap. Held Slurm job 203329 was
submitted before this flag was added; replace its script snapshot before
releasing any production job. Both RoboCasa jobs remain held for review of
performance; running LIBERO was not changed.

### H100 submission (2026-09-22)

After the previous jobs were cancelled, LIBERO was resubmitted as job 203406
with its existing premium recipe (2 GPUs, global 32, workers 8/rank), resuming
the complete production checkpoint 21000. RoboCasa was submitted as job
203408 using `finetune_rldx1_robocasa_image_h100_2gpu.sh`: H100 partition,
2 GPUs, global 64, accumulation 1, sharded loader, workers 32/rank, and the
verified production checkpoint 3650. Both retain a 60000-step schedule.
CPU allocation is left entirely to the partition defaults as requested.
The cluster requires MODEL_OUTPUT_DIR in the submission environment and a
job name of at least 50 characters; it rejects explicit cpus-per-task.

The disposable PCIe batch-scaling probe completed before cancellation:

| Samples/rank | Global batch | Seconds/update | Samples/second |
|---|---|---|---|
| 8 | 16 | 1.092128 | 14.6503 |
| 16 | 32 | 1.396660 | 22.9118 |
| 32 | 64 | 2.053756 | 31.1624 |

Each phase measured 30 updates after 10 warmup updates. This demonstrates
improved sample throughput with larger batches; step time does not scale
linearly with batch size. These are PCIe measurements, not H100 forecasts.
No updates from this variable-batch probe were saved into the baseline.

H100 job 203408 failed after 5m18s with CUDA out-of-memory at the first
NCCL dataset-statistics barrier, before any training update. The log confirms
2x H100 80GB but does not record free memory at failure; batch-size OOM and
NVLS/driver-specific causes are not established. Job 203529 retries the same
training configuration with H100-only early CUDA memory logging, explicit
NCCL device binding, a barrier before model setup, and NCCL INFO diagnostics.
This is a diagnostic retry, not a verified root-cause fix. LIBERO is unchanged.

The installed NCCL is 2.26.2. NVIDIA documents that versions before 2.26.5
lack automatic fallback when NUMA-dependent cuMem host allocations fail:
https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2265/user-guide/docs/troubleshooting.html#cumem-host-allocations
The RoboCasa runtime launcher now sets NCCL_CUMEM_HOST_ENABLE=0 only for the
h100 partition, using the documented /dev/shm fallback. Pending job 203529
sources this runtime launcher without cancellation or resubmission. This
addresses a plausible initialization failure in the installed version; the
original log alone does not establish host-cuMem as the root cause. Confirm
CUDA_NCCL_READY and actual training updates before reporting the issue fixed.
