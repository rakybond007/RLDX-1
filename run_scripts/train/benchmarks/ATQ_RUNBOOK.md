# ATQ label-gated MoE on RLDX-1: run book

How to reproduce the two ATQ fine-tunes on a new machine. Everything below runs
from the repository root on branch `RLDX-1-ATQ`.

| Run | Recipe | Key settings |
|---|---|---|
| RoboCasa ATQ | `run_scripts/train/benchmarks/finetune_rldx1_robocasa_atq_image_2gpu.sh` | speed 2.0, `vlm_contact` label, discrete dims `6 11`, tau 0.5, global batch 64, 60k steps, sharded loader |
| LIBERO ATQ | `run_scripts/train/benchmarks/finetune_rldx1_libero_atq_image_2gpu.sh` | speed 1.67, `conf_both` label, discrete dim `6`, tau 0.40, global batch 32, 60k steps, sharded loader |

Both use `RLWRLD/RLDX-1-PT-IMG`, one current frame per camera, action horizon 16,
legacy (summed) rotation targets and randomly initialised expert decoders, which
matches the upstream GR00T-action-quantization gpu26 runs.

## 1. Environment

```bash
git clone -b RLDX-1-ATQ https://github.com/rakybond007/RLDX-1.git && cd RLDX-1
bash run_scripts/train/setup_uv.sh          # uv sync --frozen; builds flash-attn if no wheel is cached
.venv/bin/python -m unittest tests/test_atq_moe.py   # 20 CPU tests, ~15 s
mkdir -p out models
```

Base checkpoint: either let the launcher download `RLWRLD/RLDX-1-PT-IMG` from
Hugging Face, or place/symlink it at `models/RLDX-1-PT-IMG` (needs
`model.safetensors.index.json` there).

## 2. Datasets

Base LeRobot datasets (unchanged): RoboCasa `robocasa_mg_gr00t_300` (7,200
episodes, action 12) and LIBERO `libero_gr00t_delta` (1,693 episodes, action 7).

The conf label rides as two extra action dims `[conf, valid]` baked into a copy
of each dataset. Bake on a compute node, not a login node: reading 1,693 LIBERO
parquets with embedded images was OOM-killed twice on a login-node cgroup.

```bash
export HF_TOKEN_FILE=~/.cache/huggingface/token     # token for prehj/* label repos
LABELS=/path/to/labels; DATASETS=/path/to/datasets

# RoboCasa: contact release -> sidecars -> bake (variant vlm_contact)
.venv/bin/python run_scripts/data/atq_labels/download_robocasa_ratio_labels.py \
    --out $LABELS/robocasa-ratio-labels-contact --token-file $HF_TOKEN_FILE
.venv/bin/python run_scripts/data/atq_labels/prepare_robocasa_ratio_sidecars.py \
    --dataset $DATASETS/robocasa_mg_gr00t_300 \
    --labels $LABELS/robocasa-ratio-labels-contact/labels/robocasa.parquet \
    --out $LABELS/robocasa-ratio-labels-contact/sidecar
.venv/bin/python run_scripts/data/atq_labels/validate_robocasa_ratio_sidecars.py \
    --dataset $DATASETS/robocasa_mg_gr00t_300 --sidecar $LABELS/robocasa-ratio-labels-contact/sidecar
.venv/bin/python run_scripts/data/atq_labels/materialize_conf_dataset.py \
    --src $DATASETS/robocasa_mg_gr00t_300 --sidecar $LABELS/robocasa-ratio-labels-contact/sidecar \
    --dst $DATASETS/robocasa_conf_vlm_contact --variant vlm_contact

# LIBERO: v3c release -> bake (gate conf_both, valid column contact_valid)
.venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('prehj/libero-conf-labels-v3c', repo_type='dataset', local_dir='$LABELS/libero-conf-labels-v3c', token=open('$HF_TOKEN_FILE').read().strip())"
.venv/bin/python run_scripts/data/atq_labels/materialize_conf_v7.py \
    --src $DATASETS/libero_gr00t_delta --labels $LABELS/libero-conf-labels-v3c/labels/libero_v3c.parquet \
    --dst $DATASETS/libero_confv3c_both --gate conf_both --valid-col contact_valid \
    --drop-columns observation.images.front_view,observation.images.left_wrist_view
```

Expected: RoboCasa validate prints `status ok` with observed
`[2073457, 2044657, 28800, [713083,1007406,194083,130085], 338533]`; the baked
RoboCasa dataset has 7,200 parquets and `ratio_label: {start: 12, end: 14}`;
LIBERO has 1,693 parquets and `ratio_label: {start: 7, end: 9}`.

## 3. Smoke (10 steps, 2 GPUs)

```bash
CONF_DATA_ROOT=$DATASETS SPEED=2.0 VARIANT=vlm_contact MAX_STEPS=10 SAVE_STEPS=10 RUN_NAME=smoke_robocasa_atq \
  bash run_scripts/train/benchmarks/finetune_rldx1_robocasa_atq_image_2gpu.sh
CONF_DATA_ROOT=$DATASETS SPEED=1.67 GATE=both MAX_STEPS=10 SAVE_STEPS=10 RUN_NAME=smoke_libero_atq \
  bash run_scripts/train/benchmarks/finetune_rldx1_libero_atq_image_2gpu.sh
```

A healthy log prints `[ATQ] experts=[...] horizons=[16, 8, 4, 8]` (RoboCasa) or
`[16, 9, 5, 8]` (LIBERO) and a diagnostics dict with `loss_conf` well above 0,
`loss_diag_gate_valid_frac` near 1.0 and four `loss_main/m8/m4/n8` values.

## 4. Full runs

```bash
CONF_DATA_ROOT=$DATASETS SPEED=2.0 VARIANT=vlm_contact MODEL_OUTPUT_DIR=/path/to/ckpts \
  sbatch run_scripts/train/benchmarks/finetune_rldx1_robocasa_atq_image_2gpu.sh
CONF_DATA_ROOT=$DATASETS SPEED=1.67 GATE=both MODEL_OUTPUT_DIR=/path/to/ckpts \
  sbatch run_scripts/train/benchmarks/finetune_rldx1_libero_atq_image_2gpu.sh
```

Override the `#SBATCH` partition/time with `sbatch --partition=... --time=...`
when the cluster differs. Re-submitting with the same `RUN_NAME` resumes from
the newest checkpoint under `$MODEL_OUTPUT_DIR/$RUN_NAME`.

Environment knobs common to both scripts: `SPEED`, `TAU`, `ROTATION_MERGE`
(`legacy` default, `so3` optional on RoboCasa), `INIT_EXPERTS_FROM_MAIN` (0
default), `NUM_WORKERS`, `GRAD_ACCUM`, `MAX_STEPS`, `SAVE_STEPS`, `BASE_MODEL_PATH`,
`DATA_DIR`. LIBERO adds `GATE` (`both|vlm|contact`) and `DISCRETE_DIMS` (`6`
default, `""` to sum the gripper like the upstream LIBERO run did).

## 5. Observed on the sjw_alinlab cluster (2026-09-22)

- Login-node cgroup OOM-killed the LIBERO bake; running it as a 1-GPU
  `background` job took a few minutes.
- `worker-node1001` (h100) failed every 2-GPU job at the first NCCL barrier with
  `CUDA error: out of memory` (ours and hojin2's); exclude it.
- The `standard` loader ran RoboCasa at 4.5 s/step (dataloader-bound, 8 decode
  workers at 200% CPU); the `sharded` loader with 12 workers profiled at
  2.2 s/step, which is why both recipes use it.
