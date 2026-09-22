# LIBERO image-model DemoSpeedup entropy collection

Use the completed 60k image checkpoint to query **every frame** of the original
LIBERO demonstrations. This is an RLDX flow-head adaptation of the released
[DemoSpeedup](https://arxiv.org/html/2506.05064v1) simulation entropy procedure:
10 stochastic chunks per observation, Gaussian KDE bandwidth 1, and causal
temporal pooling of predictions for the same frame. Only the seven normalized
LIBERO action dimensions enter the entropy; the padded dimensions are excluded.
The native 16-step horizon and checkpoint denoising schedule remain unchanged.
The image backbone runs once per observation; the stochastic action head runs
ten samples. A fixed-RNG check compares the cached path to full model inference
before collecting data.

The requested **slow=2, fast=4** ratios are recorded in each manifest for the
subsequent labeling/retiming stage. Collection does not yet classify frames,
retime demonstrations, halve the training horizon, or launch accelerated
training. It preserves raw and temporally aggregated KDE/std estimates plus
episode-normalized aggregated entropy for that next stage.

From the repository root:

```bash
MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/checkpoints/rldx1_image_baselines \
  sbatch run_scripts/demospeedup/collect_libero.sbatch
```

32 disjoint episode shards (52–53 episodes each) run with at most two GPUs concurrently, with the
partition's default CPU allocation. Completed episodes resume without repeating
inference. Output: `outputs/demospeedup/libero_60000_slow2_fast4/`, containing
one Parquet file per episode and manifests. Source data are unchanged.
Progress lines report measured frames/sec and **per-shard** remaining time;
whole-job completion also depends on the queued shards' start times. To resume
failed shards, resubmit their array indices, e.g. `sbatch --array=3,7,12%2
run_scripts/demospeedup/collect_libero.sbatch` with the same submission environment.
Each episode is atomically published after completion; an interrupted episode
is recomputed to preserve the temporal pool and RNG sequence. Shard locks reject
duplicate concurrent writers. Existing results are checked for frame coverage,
episode identity and finite entropy values before skipping.

`.venv/bin/python run_scripts/demospeedup/plan_resume.py` validates all completed
episodes and prints the missing shard indices and a submission command, without
submitting anything. Keep the shard count fixed at 32 for this sbatch recipe.

Validation: `.venv/bin/python tests/test_demospeedup_entropy.py` checks the
released KDE formula and causal offsets, horizon eviction, and episode reset.
