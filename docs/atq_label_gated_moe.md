# ATQ label-gated variable-horizon MoE

Port of the label-gated ATQ method from `GR00T-action-quantization`
(branch `jimin-dev-label-gated`, recipe
`run_scripts/gpu26/atq_label_gated_train.sh`) onto RLDX-1's MSAT action
model. Original ATQ let a router choose freely among four experts; the
label-gated variant uses a VLM speed-confidence label so that **the label
(at inference: a conf head) decides compress-vs-fine and the router only
picks the horizon inside that group**.

Everything is off unless `--use-atq-moe` is passed. Without it the model,
processor and checkpoints are byte-identical to before.

## 1. Expert grid

| idx | name | source steps | rows at 2.0x | rows at 2.5x | group |
|---:|---|---:|---|---|---|
| 0 | `main` | 16 | 16 | 16 | fine |
| 1 | `m8` | 16 | `[2]*8` → 8 | `[2,3,2,3,2,3]` → 6 (15 covered) | compressed |
| 2 | `m4` | 8 | `[2]*4` → 4 | `[2,3,3]` → 3 | compressed |
| 3 | `n8` | 8 | 8 | 8 | fine |

The MSAT body, state encoder, action encoder and action position embedding
are shared. Each expert runs the body at its own horizon (MSAT's RoPE and
the position embedding are indexed by sequence length, nothing is
hardcoded to 16) and decodes with a private `CategorySpecificMLP`
(`action_decoder`, `m8_action_decoder`, `m4_action_decoder`,
`n8_action_decoder`). New decoders are initialised from the pretrained main
decoder (`--atq-init-experts-from-main`, default on) — upstream traced its
early router collapse onto `main` to the random-init asymmetry.

**Compression rule** (`rldx/model/modules/action_model/atq.py`): continuous
dims are summed over a block, `--atq-discrete-action-dims` take the block's
last value, and a block is valid only if every source step is valid.
`--atq-action-merge-reduction last` switches every dim to last-of-block
(absolute-action spaces). `--atq-speed` selects the plan (2.0, 2.5, 1.67,
3.0 for H=16); `--atq-block-plan-full/--atq-block-plan-half` override it.

**Rotation deltas.** `--atq-rotation-merge legacy` sums the axis-angle
deltas (upstream default). `so3` composes them on SO(3): the pipeline reads
the rotation key's normaliser bounds (q01/q99) after the dataset has set its
statistics, builds a spec (`atq_rotation_merge_spec`, persisted in
`config.json`) and the merged rotation is written back in the *1x*
normalised parametrisation. `--atq-rotation-controller-scale` is radians per
raw action unit and must match the OSC controller (0.5 for RoboCasa).

## 2. Router, conf head, loss

Router input is `mean(cog tokens) ⊕ mean(state token)` (4096 + 1536), a
2-layer MLP → 4 logits. Training probabilities:

```
p = softmax(logits / atq_router_temp)        # 0.5
p = clamp(p, min=atq_min_prob); p /= p.sum()   # 0.05 keeps every expert alive
```

The conf head is a second 2-layer MLP on the same (detached) input → one
scalar, regressed onto the conf label with masked MSE (tail frames have
`valid = 0`). Total loss:

```
total = Σ_i p_i · loss_i                                  # soft mixture
      + atq_conf_loss_coef · conf_loss                    # 0.1
      + warmup · atq_balance_weight  · Σ (mean_b p_i − 1/4)²   # 0.05
      + warmup · atq_supervise_weight · KL(p ‖ softmax(−loss/atq_target_temp))  # 0.1, 0.3
warmup = min(1, step / atq_router_warmup_steps)           # 5000
```

`loss_i` is each expert's flow-matching velocity MSE at its own horizon
(fresh noise and τ per expert). **There is no group gate during training**
— gating on the label there would bake `tau` into the weights and make it
impossible to sweep at eval.

## 3. Inference

`RLDXActionModel.get_action` → `_atq_get_action`:

1. `probs = softmax(logits / atq_inference_temp)`
2. if `atq_label_gated`: `conf_pred = conf_readout(...)`, `compress = conf_pred >= atq_conf_threshold`,
   probs of the other group are zeroed and the rest renormalised
   (softmax ratios inside a group are preserved).
3. argmax (or multinomial with `--atq-inference-stochastic`) picks the expert;
   only that expert is denoised, at its horizon (Euler, `num_inference_timesteps`).
4. Returns `action_pred` zero-padded to 16 rows plus `atq_picked`,
   `atq_probs`, `atq_horizon`, `atq_conf`.

`tau` is an eval knob: change `model.config.atq_conf_threshold` on the
loaded model to sweep it on one checkpoint.

### Decoding compressed rows — why the processor needs the block sizes

RLDX min-max normalises `x' = a·x + b` with `b = −(max+min)/(max−min)`. A
row that is the sum of `k` normalised steps equals `a·Σx + k·b`, so the
plain inverse (which also clips to [−1, 1]) is wrong for compressed
experts. `RLDXProcessor.decode_action(..., block_sizes=..., sum_exempt_dims=...)`
uses the exact inverse `Σx = (y − k·b)/a` without clipping; discrete
last-of-block dims and SO(3)-composed rotation dims are exempt (`k = 1`).
`PolicyRuntime._decode_atq` does this automatically and returns the picked
expert, horizon, block sizes and probabilities in the info dict. Each row of
a compressed chunk is meant to be executed as **one** environment step.

Mixed expert picks inside one batch are rejected — serve ATQ with one
session per request.

## 4. Data: the conf label carrier

The label rides as two extra action dims `[conf, valid]` baked into a copy
of the dataset (`run_scripts/data/atq_labels/`, action 12 → 14,
`modality.json` gains `ratio_label: {start: 12, end: 14}`). The modality
config lists it as an action key (`rldx/configs/data/robocasa_conf_config.py`)
so the LeRobot loader slices it like any joint group.

It is stripped in `RLDXProcessor.__call__` **before** `state_action_processor.apply`
— the same stage as GR00T's `ConfGR00TTransform` — and re-emitted as
`conf_target` / `conf_valid`. `StateActionProcessor.skip_action_keys`
excludes it from normalisation, unnormalisation and action-dim counting, so
min-max never remaps the label and `tau` keeps its meaning. It never enters
the concatenated action tensor, the loss mask or `decode_action`.

Variants (`--dataset-path` picks one, there is no code branch):

| dataset | target |
|---|---|
| `robocasa_conf_vlm` | `conf` |
| `robocasa_conf_vlm_contact` | `conf · (1 − fixed)` |
| `robocasa_conf_contact` | `1 − fixed` |

## 5. Training

```bash
SPEED=2.0 VARIANT=vlm_contact ROTATION_MERGE=so3 TAU=0.5 \
  sbatch run_scripts/train/benchmarks/finetune_rldx1_robocasa_atq_image_2gpu.sh
```

Key flags (all `RLDXConfig` fields prefixed `atq_`, copied from the CLI and
persisted in `config.json`):

| Flag | Default | What |
|---|---|---|
| `--use-atq-moe` | off | build the 4-expert head |
| `--atq-speed` | 2.0 | compressed-group speed / block plan |
| `--atq-discrete-action-dims` | `[]` | last-of-block dims (RoboCasa `6 11`) |
| `--atq-rotation-merge` | legacy | `so3` composes rotation deltas |
| `--atq-rotation-key` | `end_effector_rotation` | 3-dim rotation action key (so3) |
| `--atq-label-gated` | on | conf head + inference gate; off = free ATQ routing |
| `--atq-conf-carrier-key` | `ratio_label` | action key carrying `[conf, valid]` |
| `--atq-conf-threshold` | 0.5 | inference `tau` (sweepable on the checkpoint) |
| `--atq-conf-loss-coef` | 0.1 | conf regression weight |
| `--atq-min-prob` | 0.05 | anti-collapse floor |
| `--atq-router-warmup-steps` | 5000 | balance / KL warmup |

Constraints enforced at assembly: one embodiment per run, no training-time
RTC, no physics stream, carrier key present in the action modality keys when
label-gated, rotation key present when `so3`.

## 6. Log keys to watch

`RLDXTrainer` logs every scalar output prefixed `loss_` / `atq_`:

| key | healthy |
|---|---|
| `loss_conf` | conf MSE — **exactly 0.0 means the label never reached the model** |
| `loss_diag_conf_mae`, `loss_diag_conf_mean`, `loss_diag_conf_target_mean` | read alongside `tau` |
| `loss_diag_gate_valid_frac` | slightly below 1.0 (4 tail frames per episode) |
| `loss_main`, `loss_m8`, `loss_m4`, `loss_n8` | per-expert flow-matching loss |
| `loss_diag_router_p_*` | router probability per expert |
| `loss_diag_router_comp_split` | m8 vs m4 inside the compressed group; 0.5 = no preference |
| `loss_diag_router_comp_flat_frac` | fraction of samples with both compressed experts on the `min_prob` floor — judge after warmup (5k steps), not before |
| `atq_warmup` | 0 → 1 over `atq_router_warmup_steps` |

## 7. Tests

```bash
.venv/bin/python -m unittest tests/test_atq_moe.py -v
```

CPU-only: block plans, compression (sum / last / discrete / partial-block
mask), group mask, SO(3) composition against a Rodrigues oracle, and the
k-corrected unnormalisation round trip.

## 8. Not ported (yet)

- The RoboCasa rollout client that consumes a variable-length chunk before
  replanning (`rldx/eval/rollout_policy.py` executes a fixed
  `n_action_steps`). The model / policy side already returns the horizon.
- `moe_uniform_router`, 2-expert mode, attention-pooled router inputs and
  the EMA-normalised KL target — upstream ablations, off in the recipe.
