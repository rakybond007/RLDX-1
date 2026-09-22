#!/usr/bin/env python3
"""Bake a v7 conf label into a copy of the RoboCasa parquets.

Differences from the v1 sidecar this replaces (materialize_conf_dataset.py):

  * one parquet for the whole benchmark, keyed by (episode_index, frame_index),
    instead of a sidecar per episode;
  * the gate is a finished column -- `conf_both` already zeroes the contact
    boundary -- so there is no conf x (1 - fixed) to reconstruct here;
  * it is labelled at **stride 4**, so the carrier has to be interpolated onto
    every frame.  The producer measured linear interpolation at conf corr 0.966
    / binary recall 95.1%, which is the reconstruction this uses.

Each episode's labels run 0, 4, 8, ... up to 4-7 frames short of the end: a
window needs its future frames to be scored.  Those trailing frames get
valid=0 and a zero target, the same convention the v1 bake used, so the loss
mask keeps them out rather than teaching a made-up value.

The carrier is appended to the flat `action` array (D -> D+2) because
modality.json slices that one array by index; that is the format the stock
loader already understands, and it keeps the custom Dataset subclass -- the
source of both previous label bugs -- out of the picture.
"""
import argparse, json, os, shutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RATIO_COLUMN = "ratio_label"
GATES = ("conf_both", "conf_vlm", "conf_contact", "conf")  # v22 ships a single "conf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="base LeRobot dataset root")
    ap.add_argument("--labels", required=True, help="v7 labels parquet")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--gate", default="conf_both", choices=GATES)
    ap.add_argument("--missing-episodes", default="fail",
                    choices=["fail", "mask", "skip"],
                    help="what to do with an episode the label file does not cover.  "
                         "v22 labels 6,993 of RoboCasa's 7,200.  'mask' keeps the "
                         "episode with valid=0 everywhere, so the action decoders "
                         "still train on it and only the conf head is masked out; "
                         "'skip' leaves it out of the copy entirely; 'fail' is the "
                         "old behaviour and the right default for a release that "
                         "claims full coverage.")
    ap.add_argument("--drop-columns", default="",
                    help="comma-separated parquet columns to leave out of the copy.  "
                         "LIBERO embeds both camera streams in the parquet (26.7 MiB "
                         "per episode) even though the loader reads video from the "
                         "mp4s named by info.json's video_path, so copying them "
                         "inflates the bake from ~200 MB to 54 GiB for nothing.")
    ap.add_argument("--valid-col", default="",
                    help="label column that marks a usable frame.  RoboCasa v7 has "
                         "none (validity is 'up to the last labelled frame'); LIBERO "
                         "v3c carries contact_valid.")
    a = ap.parse_args()

    cols = ["episode_index", "frame_index", a.gate] + ([a.valid_col] if a.valid_col else [])
    lab = pd.read_parquet(a.labels, columns=cols)
    lab = lab.sort_values(["episode_index", "frame_index"])
    by_ep = {e: (g.frame_index.to_numpy(np.int64), g[a.gate].to_numpy(np.float32),
                 g[a.valid_col].to_numpy(np.float32) if a.valid_col else None)
             for e, g in lab.groupby("episode_index", sort=False)}
    print(f"[labels] {len(lab):,} rows over {len(by_ep)} episodes, gate={a.gate}")

    drop = {c.strip() for c in a.drop_columns.split(",") if c.strip()}
    if drop:
        print(f"[data] dropping columns: {sorted(drop)}")
    os.makedirs(a.dst, exist_ok=True)
    if os.path.exists(f"{a.dst}/meta"):
        shutil.rmtree(f"{a.dst}/meta")
    shutil.copytree(f"{a.src}/meta", f"{a.dst}/meta")
    vlink = f"{a.dst}/videos"
    if os.path.islink(vlink):
        os.remove(vlink)
    elif os.path.exists(vlink):
        shutil.rmtree(vlink)
    os.symlink(f"{a.src}/videos", vlink)

    n_ep = n_rows = n_tail = n_skipped = n_masked = 0
    base_dim = None
    for root, _, files in os.walk(f"{a.src}/data"):
        for fn in sorted(f for f in files if f.endswith(".parquet")):
            src_p = os.path.join(root, fn)
            dst_p = os.path.join(a.dst, "data", os.path.relpath(src_p, f"{a.src}/data"))
            os.makedirs(os.path.dirname(dst_p), exist_ok=True)
            ep = int(fn.split("_")[-1].split(".")[0])
            base = pq.read_table(src_p)
            if drop:
                missing = drop - set(base.schema.names)
                if missing:
                    raise ValueError(f"--drop-columns names absent columns: {sorted(missing)}")
                base = base.select([c for c in base.schema.names if c not in drop])
            T = base.num_rows
            if ep not in by_ep:
                if a.missing_episodes == "skip":
                    n_skipped += 1
                    continue
                if a.missing_episodes == "fail":
                    raise ValueError(f"ep{ep}: no labels "
                                     f"(pass --missing-episodes mask|skip if the "
                                     f"release does not cover every episode)")
                f_lab, c_lab, v_lab = None, None, None
            else:
                f_lab, c_lab, v_lab = by_ep[ep]
            frames = np.arange(T, dtype=np.int64)
            if f_lab is None:
                # Unlabelled episode under --missing-episodes mask: a zero target
                # behind a zero mask teaches the conf head nothing, while the
                # action decoders still see the trajectory.
                tgt = np.zeros(T, dtype=np.float32)
                valid = np.zeros(T, dtype=np.float32)
                n_masked += T
            else:
                if f_lab[-1] >= T:
                    raise ValueError(f"ep{ep}: label frame {f_lab[-1]} past episode length {T}")
                # Equal-spaced labels make np.interp the overlap-weighted average
                # the v22 release asks for, and it returns the node value exactly
                # at a labelled frame, so a tau picked on the labels still holds.
                if len(f_lab) == T and f_lab[-1] == T - 1:
                    tgt = c_lab.astype(np.float32)
                    valid = (v_lab if v_lab is not None
                             else np.ones(T, dtype=np.float32)).astype(np.float32)
                else:
                    tgt = np.interp(frames, f_lab, c_lab).astype(np.float32)
                    # Only the span the labels actually bracket is supervised.
                    # np.interp holds the edge value outside that span, which
                    # would invent a target: v22 has one episode whose labels
                    # start at frame 16 and episodes whose labels stop up to 227
                    # frames before the end.  Interior gaps (18 episodes have a
                    # 32-frame step where a block was dropped) are genuinely
                    # bracketed and stay supervised, just coarser.
                    valid = ((frames >= f_lab[0]) & (frames <= f_lab[-1])).astype(np.float32)
                    if v_lab is not None:
                        valid = valid * np.interp(frames, f_lab, v_lab).astype(np.float32)
                        valid = (valid > 0.999).astype(np.float32)
            tgt = np.where(valid > 0.5, tgt, 0.0).astype(np.float32)
            if not np.isfinite(tgt).all() or tgt.min() < 0.0 or tgt.max() > 1.0:
                raise ValueError(f"ep{ep}: target out of range {tgt.min()}..{tgt.max()}")
            n_tail += int((valid < 0.5).sum())

            act = np.asarray(base.column("action").to_pylist(), dtype=np.float32)
            if act.ndim != 2:
                raise ValueError(f"ep{ep}: action is {act.shape}, expected (T, D)")
            if base_dim is None:
                base_dim = act.shape[1]
            elif act.shape[1] != base_dim:
                raise ValueError(f"ep{ep}: action dim {act.shape[1]} != {base_dim}")
            wide = np.concatenate([act, np.stack([tgt, valid], 1)], axis=1).astype(np.float32)
            out = base.drop(["action"]).append_column(
                "action", pa.array([r.tolist() for r in wide],
                                   type=pa.list_(pa.float32(), wide.shape[1])))
            pq.write_table(out, dst_p)
            n_ep += 1; n_rows += T
            if n_ep % 1000 == 0:
                print(f"  {n_ep} episodes", flush=True)
    print(f"[data] {n_ep} episodes / {n_rows:,} frames "
          f"({n_tail:,} tail frames masked out"
          + (f", {n_skipped} episodes skipped" if n_skipped else "")
          + (f", {n_masked:,} frames from unlabelled episodes masked" if n_masked else "")
          + ")")

    lo, hi = base_dim, base_dim + 2
    mp = f"{a.dst}/meta/modality.json"
    mod = json.load(open(mp))
    if max(v["end"] for v in mod["action"].values()) != base_dim:
        raise ValueError("modality.json action slices do not end at the action width")
    mod["action"][RATIO_COLUMN] = {"start": lo, "end": hi, "absolute": True,
                                   "dtype": "float32"}
    json.dump(mod, open(mp, "w"), indent=2)

    # The slices index into these arrays, so they have to grow too.  The carrier
    # is deliberately absent from action_normalization_modes, so the values are
    # never applied -- identity entries keep the shapes honest without
    # pretending the conf scale was measured.
    for name in ("stats.json", "relative_stats.json", "stats_gr00t.json"):
        sp = f"{a.dst}/meta/{name}"
        if not os.path.exists(sp):
            continue
        st = json.load(open(sp))
        blk = st.get("action")
        if not isinstance(blk, dict):
            continue
        for key, fill in (("mean", 0.0), ("std", 1.0), ("min", 0.0), ("max", 1.0),
                          ("q01", 0.0), ("q10", 0.0), ("q50", 0.0),
                          ("q90", 1.0), ("q99", 1.0)):
            if key in blk and isinstance(blk[key], list):
                if len(blk[key]) != base_dim:
                    raise ValueError(f"{name}: action.{key} is {len(blk[key])}, want {base_dim}")
                blk[key] = list(blk[key]) + [fill, fill]
        # The loader slices EVERY stat by the modality range, so a scalar count
        # raises IndexError on the second dim.
        if "count" in blk and isinstance(blk["count"], list):
            blk["count"] = list(blk["count"]) + [blk["count"][-1]] * 2 \
                if len(blk["count"]) == base_dim else [blk["count"][0]] * hi
        json.dump(st, open(sp, "w"))

    info_p = f"{a.dst}/meta/info.json"
    info = json.load(open(info_p))
    info["features"]["action"]["shape"] = [hi]  # the flat action array grew by the carrier
    info["conf_variant"] = a.gate
    info["label_release"] = os.path.basename(os.path.dirname(os.path.dirname(a.labels)))
    info["label_gate_column"] = a.gate
    info["label_valid_column"] = a.valid_col or None
    if drop:
        info["dropped_columns"] = sorted(drop)
    json.dump(info, open(info_p, "w"), indent=2)
    print(f"[done] ratio_label at {lo}:{hi}  ->  {a.dst}")


if __name__ == "__main__":
    main()
