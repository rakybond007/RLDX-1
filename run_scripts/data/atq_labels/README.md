# ATQ conf-label dataset for RLDX-1 (RoboCasa)

Bakes the VLM speed-confidence label (`prehj/robocasa-ratio-labels-contact`)
into a copy of the RoboCasa LeRobot dataset as two extra action dims
`[conf, valid]`. The four scripts are copied unchanged from
`GR00T-action-quantization/scripts` (branch `jimin-dev-ATQ-finalized-only`);
they operate on the LeRobot layout only and need `numpy`, `pyarrow`,
`huggingface_hub`.

RLDX's loader reads `meta/modality.json` slices exactly like GR00T's, so the
baked dataset works with `rldx/configs/data/robocasa_conf_config.py` without
any loader change. The processor strips the carrier before normalisation
(`conf_carrier_key`), so the label is never min-max remapped and `tau` keeps
its meaning.

## Steps (run on the cluster that mounts the dataset)

```bash
BASE=/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300
LABEL=/path/with/space/robocasa-ratio-labels-contact
OUT_ROOT=/path/with/space   # variant datasets land at $OUT_ROOT/robocasa_conf_<variant>

# 1) labels (HF token in a file — never on the command line)
.venv/bin/python run_scripts/data/atq_labels/download_robocasa_ratio_labels.py \
    --out "$LABEL" --token-file /path/to/hf_token.txt

# 2) per-episode sidecars (joins on episode_index + row position)
.venv/bin/python run_scripts/data/atq_labels/prepare_robocasa_ratio_sidecars.py \
    --dataset "$BASE" --labels "$LABEL/labels/robocasa.parquet" --out "$LABEL/sidecar"

# 3) validate — must print status ok with
#    observed [2073457, 2044657, 28800, [713083,1007406,194083,130085], 338533]
.venv/bin/python run_scripts/data/atq_labels/validate_robocasa_ratio_sidecars.py \
    --dataset "$BASE" --sidecar "$LABEL/sidecar"

# 4) bake one dataset per variant (parquet copy 1.2 GB each; videos symlinked)
for V in vlm vlm_contact contact; do
  .venv/bin/python run_scripts/data/atq_labels/materialize_conf_dataset.py \
      --src "$BASE" --sidecar "$LABEL/sidecar" --dst "$OUT_ROOT/robocasa_conf_$V" --variant $V
done
```

| variant | target | meaning |
|---|---|---|
| `vlm` | `conf` | VLM judgement only |
| `vlm_contact` | `conf * (1 - fixed)` | contact frames forced to 0 |
| `contact` | `1 - fixed` | contact flag only |

All three are "higher = more compressible", so the eval rule
`conf_pred >= tau -> compressed group` covers all of them.

## What the bake changes

| | |
|---|---|
| `data/` | 7,200 parquets copied; flat `action` grows 12 -> 14 |
| `videos/` | symlink to the base (34 GB, unchanged) |
| `meta/modality.json` | `action.ratio_label: {start: 12, end: 14}` |
| `meta/stats.json` | action stat arrays extended 12 -> 14 with identity values (never applied) |
| `meta/info.json` | `features.action.shape = [14]`, `conf_variant` recorded |

## Checks after baking

- `meta/modality.json` action slices: pos `[0,3)`, rot `[3,6)`, gripper `[6,7)`,
  base_motion `[7,11)`, control_mode `[11,12)`, ratio_label `[12,14)`. The
  training script passes `--atq-discrete-action-dims 6 11`; if your slices
  differ, change the flag.
- RLDX's stats check requires `q01`/`q99` for every float feature; the bake
  extends them, so `generate_stats` must NOT regenerate (it would print
  "Generating stats" and overwrite the file — a sign the bake was incomplete).
- Tail 4 frames per episode have `valid = 0` and `conf = 0`; the processor
  rejects any other invalid row.
