# LIBERO

Single-arm tabletop manipulation on a Franka Panda robot with a parallel
gripper (40 tasks across Spatial / Object / Goal / Long suites).

| Field | Value |
|---|---|
| Embodiment tag | `GENERAL_EMBODIMENT` |
| HuggingFace checkpoint | [`RLWRLD/RLDX-1-FT-LIBERO`](https://huggingface.co/RLWRLD/RLDX-1-FT-LIBERO) |
| Reported success rate | 97.4 % (LIBERO Avg) / 84.3 % (LIBERO-Plus) |
| Simulator venv | `rldx/eval/sim/LIBERO/libero_uv/.venv` |

## 1. Setup (one-time)

```bash
bash run_scripts/eval/libero/setup_libero.sh
```

Builds an isolated `uv` venv for LIBERO and downloads task assets.

## 2. Fine-tune from RLDX-1-PT

```bash
DATA_DIR=/path/to/libero_delta \
bash run_scripts/train/benchmarks/finetune_rldx1_libero.sh
```

Defaults to `RLWRLD/RLDX-1-PT` as the base; override with
`BASE_MODEL_PATH=...`.

## 3. Run evaluation

```bash
bash run_scripts/eval/libero/eval_libero.sh \
    libero_release \
    RLWRLD/RLDX-1-FT-LIBERO
```

Arguments: `<run_label>` (output dir name), `<MODEL_PATH>` (HF repo or
local checkpoint). Outputs land in
`output_final/libero/<run_label>/<suite>/<task>/`.

## Image baseline at 60k, replan 5

`setup_image_eval.sh` creates `.venv-libero-eval` with isolated simulator
packages. Its default base interpreter is the local `libero310` environment;
set `LIBERO_BASE_PYTHON` to override. Training `.venv` stays unchanged.
LIBERO assets/configuration are scoped to this checkout using
`LIBERO_CONFIG_PATH=.cache/libero-eval`.

```bash
bash run_scripts/eval/libero/setup_image_eval.sh
MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/checkpoints/rldx1_image_baselines \
  sbatch run_scripts/eval/libero/eval_image_baseline.sbatch
```

The array covers Spatial/Object/Goal/Long (one GPU per suite, at most two
concurrent jobs), 10 tasks/suite and 50 official initial states/task, seed 7.
Episode limits are 220/280/300/520 respectively, following the GR00T reference
protocol; ten settling actions are excluded from those limits. Every policy
request executes exactly five actions from the returned 16-action chunk.
Physics, memory, motion and RTC remain disabled in the image checkpoint.

The client uses RLDX's existing observation mapping and gripper conversion.
Per-episode JSONL records success, step count, initial-state ID and policy-call
count. Each task's first rollout is recorded as video; each suite emits
`summary.json`. Outputs are under
`output_final/libero/image_60000_replan5_<array_job_id>/<suite>/`.
A failed rollout raises an error rather than being counted as a success.

The Hugging Face inference artifact is `prehj/RLDX-1-IMG-LIBERO-60k` (private).
`run_scripts/publish/upload_libero_image.py` uploads model/processor/configuration
without printing its token or uploading DeepSpeed optimizer states. Local
training checkpoints retain the full resumable state.
