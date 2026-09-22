#!/usr/bin/env python3
"""Bake the conf label into a copy of the RoboCasa parquets.

Why this exists: merging the label at read time needed a custom Dataset
subclass, and that subclass was where both label bugs came from -- a stale
"already converted" guard that could hand class indices back as conf, and (in
the ATQ port) a carrier that never reached the action tensor at all.  Neither
showed up in a 5-step smoke.

Baking removes the subclass entirely.  Only `data/` is copied (1.2 GB, 7200
files); `videos/` is 34 GB and is symlinked, since nothing about it changes.
Each variant gets its own dataset, so the three of them share one data config
and differ only by path -- no conf_variant branching anywhere in the code.

The label semantics are documented in
GR00T-action-quantization/docs/LABEL_DATASET_MERGE.md; the three that decide
this script are: conf is NaN on the four unlabelled tail frames, conf is scored
independently of the contact override, and fixed=1 always means ratio 1.0.
"""
import argparse, json, os, shutil, sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VARIANTS = ("vlm", "vlm_contact", "contact")
RATIO_COLUMN = "ratio_label"


def build_target(conf, fixed, valid, variant):
    conf = np.nan_to_num(np.asarray(conf, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    fixed = np.asarray(fixed, dtype=np.float32)
    if not np.isin(fixed, (0.0, 1.0)).all():
        raise ValueError("ratio_fixed is not binary")
    if variant == "vlm":
        t = conf
    elif variant == "vlm_contact":
        t = conf * (1.0 - fixed)
    else:
        t = 1.0 - fixed
    t = np.where(np.asarray(valid) > 0.5, t, 0.0).astype(np.float32)
    if not np.isfinite(t).all() or t.min() < 0.0 or t.max() > 1.0:
        raise ValueError(f"target out of range: {t.min()}..{t.max()}")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="base LeRobot dataset root")
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    a = ap.parse_args()

    os.makedirs(a.dst, exist_ok=True)
    # meta is small and must be a real copy: info.json gains the new feature.
    if os.path.exists(f"{a.dst}/meta"):
        shutil.rmtree(f"{a.dst}/meta")
    shutil.copytree(f"{a.src}/meta", f"{a.dst}/meta")
    # videos are 34 GB and unchanged -> symlink, never copy.
    vlink = f"{a.dst}/videos"
    if os.path.islink(vlink) or os.path.exists(vlink):
        os.remove(vlink) if os.path.islink(vlink) else shutil.rmtree(vlink)
    os.symlink(f"{a.src}/videos", vlink)

    n_ep = n_rows = 0
    for root, _, files in os.walk(f"{a.src}/data"):
        for fn in sorted(f for f in files if f.endswith(".parquet")):
            src_p = os.path.join(root, fn)
            rel = os.path.relpath(src_p, f"{a.src}/data")
            dst_p = os.path.join(a.dst, "data", rel)
            os.makedirs(os.path.dirname(dst_p), exist_ok=True)
            ep = int(fn.split("_")[-1].split(".")[0])
            base = pq.read_table(src_p)
            side = pq.read_table(f"{a.sidecar}/episode_{ep:06d}.parquet",
                                 columns=["episode_index", "frame_index",
                                          RATIO_COLUMN, "ratio_conf", "ratio_fixed"])
            if side.num_rows != base.num_rows:
                raise ValueError(f"ep{ep}: sidecar {side.num_rows} vs base {base.num_rows}")
            if not np.all(np.asarray(side.column("episode_index").to_pylist()) == ep):
                raise ValueError(f"ep{ep}: sidecar episode identity mismatch")
            if not np.array_equal(np.asarray(side.column("frame_index").to_pylist()),
                                  np.arange(base.num_rows)):
                raise ValueError(f"ep{ep}: sidecar frame_index is not 0..n-1")
            valid = np.asarray(side.column(RATIO_COLUMN).to_pylist(), dtype=np.float32)[:, 1]
            tgt = build_target(side.column("ratio_conf").to_pylist(),
                               side.column("ratio_fixed").to_pylist(), valid, a.variant)
            # modality.json slices one flat `action` array by index, so the
            # carrier is appended to that array rather than added as its own
            # column -- that is the format the stock loader already understands.
            act = np.asarray(base.column("action").to_pylist(), dtype=np.float32)
            if act.ndim != 2:
                raise ValueError(f"ep{ep}: action is {act.shape}, expected (T, D)")
            wide = np.concatenate(
                [act, np.stack([tgt, valid], axis=1).astype(np.float32)], axis=1)
            out = base.drop(["action"]).append_column(
                "action", pa.array([r.tolist() for r in wide],
                                   type=pa.list_(pa.float32(), wide.shape[1])))
            pq.write_table(out, dst_p)
            if n_ep == 0:
                globals()["_BASE_DIM"] = act.shape[1]
            n_ep += 1; n_rows += base.num_rows
    print(f"wrote {n_ep} episodes / {n_rows} frames -> {a.dst}")

    base_dim = globals()["_BASE_DIM"]
    lo, hi = base_dim, base_dim + 2

    # modality.json: name the two new slots so the loader can find them.
    mp = f"{a.dst}/meta/modality.json"
    mod = json.load(open(mp))
    if max(v["end"] for v in mod["action"].values()) != base_dim:
        raise ValueError("modality.json action slices do not end at the action width")
    mod["action"][RATIO_COLUMN] = {"start": lo, "end": hi, "absolute": True,
                                   "dtype": "float32"}
    json.dump(mod, open(mp, "w"), indent=2)

    # stats: the slices index into these arrays, so they have to grow too.
    # The carrier is deliberately absent from action_normalization_modes, so
    # these values are never applied -- identity entries keep the shapes honest
    # without pretending the conf scale was measured.
    for name in ("stats.json", "relative_stats.json", "stats_gr00t.json"):
        sp = f"{a.dst}/meta/{name}"
        if not os.path.exists(sp):
            continue
        st = json.load(open(sp))
        tgt_block = st.get("action")
        if not isinstance(tgt_block, dict):
            continue
        for key, fill in (("mean", 0.0), ("std", 1.0), ("min", 0.0), ("max", 1.0),
                          ("q01", 0.0), ("q99", 1.0)):
            if key in tgt_block and isinstance(tgt_block[key], list):
                if len(tgt_block[key]) != base_dim:
                    raise ValueError(f"{name}: action.{key} is {len(tgt_block[key])}, "
                                     f"expected {base_dim}")
                tgt_block[key] = list(tgt_block[key]) + [fill, fill]
        json.dump(st, open(sp, "w"), indent=2)
        print(f"  extended {name} action stats {base_dim} -> {hi}")

    info_p = f"{a.dst}/meta/info.json"
    info = json.load(open(info_p))
    info["features"]["action"]["shape"] = [hi]
    info["conf_variant"] = a.variant
    json.dump(info, open(info_p, "w"), indent=2)
    print(f"action {base_dim} -> {hi}, {RATIO_COLUMN} at [{lo},{hi}) (variant={a.variant})")


if __name__ == "__main__":
    main()
