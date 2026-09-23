# RoboCasa Kitchen (24 tasks)

Single-arm kitchen manipulation on a mobile Panda manipulator
(`PandaOmron`). 24 tasks: pick-and-place, doors, drawers, appliances.
Two fixed external cameras (left/right) + one wrist camera, 256×256.

| Field | Value |
|---|---|
| Embodiment tag | `GENERAL_EMBODIMENT` |
| HuggingFace checkpoint | [`RLWRLD/RLDX-1-FT-ROBOCASA`](https://huggingface.co/RLWRLD/RLDX-1-FT-ROBOCASA) |
| Reported success rate | 70.6 % |
| Simulator venv | `rldx/eval/sim/robocasa/robocasa_uv/.venv` |

## 1. Setup (one-time)

```bash
bash run_scripts/eval/robocasa_kitchen/setup_robocasa.sh
```

Builds the simulator venv and applies patches under
`run_scripts/eval/robocasa_kitchen/patches/` (notably
`seed_clamp_64bit.patch` to fix `gymnasium 0.29` 64-bit seed forwarding).

## 2. Fine-tune from RLDX-1-PT

300-demo recipe — matches the technical-report number:

```bash
DATA_DIR=/path/to/robocasa_mg_gr00t_300 \
bash run_scripts/train/benchmarks/finetune_rldx1_robocasa.sh
```

## 3. Run evaluation

```bash
bash run_scripts/eval/robocasa_kitchen/eval_robocasa.sh \
    RLWRLD/RLDX-1-FT-ROBOCASA
```

Parallel 4-GPU runner that shards the 24 tasks across local GPUs and runs
50 episodes per task. Outputs land in
`output_final/robocasa/<ckpt>/<task>/`.

## Image 60k baseline from `prehj`

The image checkpoint at
[`prehj/rldx1-robocasa-img-gb64-60k-baseline`](https://huggingface.co/prehj/rldx1-robocasa-img-gb64-60k-baseline)
is evaluated with the same 24 tasks and 50 episodes/task as the kitchen runner
above. The existing kitchen protocol uses 720 maximum environment steps,
16 actions per policy call, and seed 42. Four array shards each run six tasks
with one model server per GPU, at most two GPUs concurrently. The checkpoint is
downloaded to `models/rldx1-robocasa-img-gb64-60k-baseline`.

The simulator uses `.venv-robocasa-eval` with Python from the existing
`robocasa_gr00t` conda environment and pinned NumPy/Gymnasium versions. Run
`bash run_scripts/eval/robocasa_kitchen/setup_image_eval.sh` once to prepare
local RoboCasa/robosuite code and link the existing kitchen assets. RoboCasa's
temporary object XML is redirected to `/tmp` with original mesh and texture
paths resolved to absolute paths; source assets remain unchanged. The source
asset location can be overridden with `ROBOCASA_ASSET_SOURCE`.
Robosuite uses JIT with its file cache disabled because its shared source path
cannot be used by Numba's disk cache. The setup is isolated under `.work`.

From the repository root, after setup and checkpoint download:

```bash
MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/checkpoints/rldx1_image_baselines \
  sbatch run_scripts/eval/robocasa_kitchen/eval_image_60k.sbatch
```

Results, videos, CSV episode records, and server logs are stored under
`output_final/robocasa/image_60000_ac16_<array_job_id>/`.
