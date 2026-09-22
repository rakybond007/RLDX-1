#!/usr/bin/env python3
"""Strict validation of prepared ratio sidecars against their base dataset."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sidecar", required=True)
    args = parser.parse_args()
    dataset, sidecar = Path(args.dataset), Path(args.sidecar)
    manifest = json.loads((sidecar / "manifest.json").read_text())
    data_pattern = manifest["dataset_data_path"]
    chunk_size = int(manifest["dataset_chunks_size"])
    counts = np.zeros(4, dtype=np.int64)
    grade_counts = np.zeros((5, 5), dtype=np.int64)
    valid_total = fixed_total = base_total = 0
    for episode in range(int(manifest["episodes"])):
        base = dataset / data_pattern.format(
            episode_chunk=episode // chunk_size, episode_index=episode,
        )
        path = sidecar / f"episode_{episode:06d}.parquet"
        nbase = pq.ParquetFile(base).metadata.num_rows
        identity = pq.read_table(base, columns=["episode_index", "index"])
        if not np.all(identity["episode_index"].to_numpy() == episode):
            raise ValueError(f"ep{episode}: base episode identity mismatch")
        base_index = identity["index"].to_numpy()
        if len(base_index) > 1 and not np.all(np.diff(base_index) == 1):
            raise ValueError(f"ep{episode}: base global index is not contiguous")
        table = pq.read_table(path)
        if len(table) != nbase:
            raise ValueError(f"ep{episode}: sidecar {len(table)} vs base {nbase}")
        if not np.all(table["episode_index"].to_numpy() == episode):
            raise ValueError(f"ep{episode}: sidecar episode identity mismatch")
        if not np.array_equal(table["frame_index"].to_numpy(), np.arange(nbase)):
            raise ValueError(f"ep{episode}: sidecar frame identity mismatch")
        target = np.asarray(table["ratio_label"].to_pylist(), dtype=np.float32)
        grade_target = np.asarray(table["grade_label"].to_pylist(), dtype=np.float32)
        valid = target[:, 1]
        if not np.isin(valid, (0, 1)).all():
            raise ValueError(f"ep{episode}: nonbinary valid")
        invalid = np.flatnonzero(valid == 0)
        expected = np.arange(max(0, nbase - 4), nbase)
        if not np.array_equal(invalid, expected):
            raise ValueError(f"ep{episode}: invalid positions {invalid.tolist()} != {expected.tolist()}")
        klass = target[valid == 1, 0]
        if not np.isin(klass, (0, 1, 2, 3)).all():
            raise ValueError(f"ep{episode}: invalid class")
        ratio = table["ratio"].to_numpy()
        grid = np.asarray(manifest["ratio_grid"], dtype=np.float32)
        if not np.allclose(ratio[valid == 1], grid[klass.astype(int)]):
            raise ValueError(f"ep{episode}: class/ratio mismatch")
        grades, grade_valid = grade_target[:, :5], grade_target[:, 5:]
        if not np.all(grade_valid[valid == 1] == 1) or not np.all(grade_valid[valid == 0] == 0):
            raise ValueError(f"ep{episode}: grade masks do not match source coverage")
        if not np.isin(grades[grade_valid.astype(bool)], (0, 1, 2, 3, 4)).all():
            raise ValueError(f"ep{episode}: invalid grade class")
        for question in range(5):
            mask = grade_valid[:, question] == 1
            grade_counts[question] += np.bincount(
                grades[mask, question].astype(int), minlength=5,
            )
        counts += np.bincount(klass.astype(int), minlength=4)
        fixed_total += int(table["ratio_fixed"].to_numpy()[valid == 1].sum())
        valid_total += int(valid.sum())
        base_total += nbase
    observed = (base_total, valid_total, base_total-valid_total, counts.tolist(), fixed_total)
    expected = (
        manifest["base_frames"], manifest["labelled_frames"], manifest["masked_frames"],
        manifest["class_counts"], manifest["fixed_count"],
    )
    if observed != expected:
        raise ValueError(f"aggregate mismatch: {observed} != {expected}")
    if grade_counts.tolist() != manifest["grade_class_counts"]:
        raise ValueError("aggregate grade distribution does not match manifest")
    print(json.dumps({"status": "ok", "observed": observed}, indent=2))


if __name__ == "__main__":
    main()
