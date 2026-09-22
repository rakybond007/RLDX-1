#!/usr/bin/env python3
"""Join the ratio release to RoboCasa and build compact per-episode sidecars.

The base release has no ``frame_index`` parquet column.  Labels use a local
0-based frame index, so the join is ``episode_index`` plus row position, checked
against ``meta/episodes.jsonl`` and every parquet row count.  The source release
omits exactly four suffix frames per episode.  Those become ``[class=0,valid=0]``;
any interior or other missing label aborts the build.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

GRID = np.asarray((1.0, 1.5, 2.0, 2.5), dtype=np.float32)
EXPECTED_TAIL = 4
SOURCE_REVISION = "c9d9fbd76a863cdac76afe8f09269c19e48bc1bf"
SOURCE_SHA256 = "ebaaaddb2055bfbaa56997c906468b12ca1f2b7918ae0d5e7b3dc2f602094534"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_episodes(dataset: Path):
    rows = {}
    with (dataset / "meta" / "episodes.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            episode = int(row["episode_index"])
            if episode in rows:
                raise ValueError(f"duplicate episode metadata: {episode}")
            rows[episode] = row
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    dataset, labels, out = Path(args.dataset), Path(args.labels), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    episodes = read_episodes(dataset)
    if sorted(episodes) != list(range(len(episodes))):
        raise ValueError("episode_index must be contiguous from zero")

    info = json.loads((dataset / "meta" / "info.json").read_text())
    chunk_size = int(info.get("chunks_size", 300))
    data_pattern = str(info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    ))
    label_sha256 = digest(labels)
    if label_sha256 != SOURCE_SHA256:
        raise ValueError(
            f"labels SHA-256 {label_sha256} is not pinned contact release {SOURCE_SHA256}"
        )

    columns = ["episode_index", "frame_index", "ratio", "conf", "fixed", "task", "A", "B", "C", "D", "E"]
    source = pq.read_table(labels, columns=columns).combine_chunks()
    ep = source["episode_index"].to_numpy().astype(np.int64, copy=False)
    frame = source["frame_index"].to_numpy().astype(np.int64, copy=False)
    ratio = source["ratio"].to_numpy().astype(np.float32, copy=False)
    conf = source["conf"].to_numpy().astype(np.float32, copy=False)
    fixed = source["fixed"].to_numpy().astype(np.int8, copy=False)
    tasks = source["task"].to_pylist()
    raw_grades = np.stack([source[name].to_numpy() for name in "ABCDE"], axis=1).astype(np.int8)
    order = np.lexsort((frame, ep))
    if not np.array_equal(order, np.arange(len(ep))):
        raise ValueError("source labels must be sorted by (episode_index, frame_index)")
    if len(ep) and (ep.min() < 0 or ep.max() >= len(episodes)):
        raise ValueError(f"source episode_index outside 0..{len(episodes)-1}")
    if len(ep) > 1 and np.any((ep[1:] == ep[:-1]) & (frame[1:] == frame[:-1])):
        raise ValueError("duplicate source label key")
    if not np.isin(ratio, GRID).all() or not np.isfinite(conf).all():
        raise ValueError("invalid ratio/conf value")
    if np.any((conf < 0) | (conf > 1)) or not np.isin(fixed, (0, 1)).all():
        raise ValueError("conf/fixed outside allowed range")
    if np.any((fixed == 1) & (ratio != 1.0)):
        raise ValueError("contact-fixed rows must have ratio=1.0")
    if not np.isin(raw_grades, (1, 2, 3, 4, 5)).all():
        raise ValueError("A-E grades must be complete integers in 1..5")
    class_index = np.abs(ratio[:, None] - GRID[None]).argmin(1).astype(np.int8)

    offsets = np.searchsorted(ep, np.arange(len(episodes) + 1), side="left")
    class_counts = np.zeros(4, dtype=np.int64)
    fixed_count = valid_count = base_count = 0
    task_names = set()
    for episode in range(len(episodes)):
        meta = episodes[episode]
        length = int(meta["length"])
        parquet = dataset / data_pattern.format(
            episode_chunk=episode // chunk_size, episode_index=episode,
        )
        actual = pq.ParquetFile(parquet).metadata.num_rows
        if actual != length:
            raise ValueError(f"ep{episode}: metadata length {length}, parquet rows {actual}")
        identity = pq.read_table(parquet, columns=["episode_index", "index"])
        base_ep = identity["episode_index"].to_numpy()
        base_index = identity["index"].to_numpy()
        if not np.all(base_ep == episode):
            raise ValueError(f"ep{episode}: parquet episode_index does not match filename")
        if len(base_index) > 1 and not np.all(np.diff(base_index) == 1):
            raise ValueError(f"ep{episode}: parquet global index is not contiguous")
        lo, hi = int(offsets[episode]), int(offsets[episode + 1])
        frames = frame[lo:hi]
        expected = np.arange(max(0, length - EXPECTED_TAIL), dtype=np.int64)
        if not np.array_equal(frames, expected):
            raise ValueError(
                f"ep{episode}: label frames are not exact 0..{length-EXPECTED_TAIL-1}; "
                f"got {frames[:3].tolist()}..{frames[-3:].tolist() if len(frames) else []}"
            )
        if not np.all(ep[lo:hi] == episode):
            raise ValueError(f"ep{episode}: source slice episode mismatch")
        task = str(tasks[lo]) if hi > lo else ""
        if any(str(value) != task for value in tasks[lo:hi]):
            raise ValueError(f"ep{episode}: source labels contain more than one task")
        expected_task = str(meta.get("tasks", ["", ""])[1])
        if task != expected_task:
            raise ValueError(f"ep{episode}: label task {task!r} != metadata task {expected_task!r}")
        task_names.add(task)

        carrier = np.zeros((length, 2), dtype=np.float32)
        carrier[:len(frames), 0] = class_index[lo:hi]
        carrier[:len(frames), 1] = 1.0
        grade_carrier = np.zeros((length, 10), dtype=np.float32)
        grade_carrier[:len(frames), :5] = raw_grades[lo:hi] - 1
        grade_carrier[:len(frames), 5:] = 1.0
        audit_ratio = np.ones(length, dtype=np.float32)
        audit_conf = np.full(length, np.nan, dtype=np.float32)
        audit_fixed = np.zeros(length, dtype=np.int8)
        audit_ratio[:len(frames)] = ratio[lo:hi]
        audit_conf[:len(frames)] = conf[lo:hi]
        audit_fixed[:len(frames)] = fixed[lo:hi]
        table = pa.table({
            "episode_index": np.full(length, episode, dtype=np.int32),
            "frame_index": np.arange(length, dtype=np.int32),
            "ratio_label": pa.array(carrier.tolist(), type=pa.list_(pa.float32(), 2)),
            "grade_label": pa.array(grade_carrier.tolist(), type=pa.list_(pa.float32(), 10)),
            "ratio": audit_ratio, "ratio_conf": audit_conf,
            "ratio_fixed": audit_fixed,
        })
        pq.write_table(table, out / f"episode_{episode:06d}.parquet", compression="zstd")
        class_counts += np.bincount(class_index[lo:hi], minlength=4)
        fixed_count += int(fixed[lo:hi].sum())
        valid_count += hi - lo
        base_count += length

    manifest = {
        "format_version": 1,
        "source_repo": "prehj/robocasa-ratio-labels-contact",
        "source_revision": SOURCE_REVISION,
        "source_file": labels.name,
        "source_sha256": label_sha256,
        "dataset_name": dataset.name,
        "dataset_episodes_sha256": digest(dataset / "meta" / "episodes.jsonl"),
        "dataset_info_sha256": digest(dataset / "meta" / "info.json"),
        "dataset_data_path": data_pattern, "dataset_chunks_size": chunk_size,
        "join": "episode_index + zero-based row position",
        "ratio_grid": GRID.astype(float).tolist(),
        "target": "ratio_label=[class_index, valid]",
        "grade_target": "grade_label=[gA,gB,gC,gD,gE,validA,validB,validC,validD,validE]",
        "grade_class_counts": [
            np.bincount(raw_grades[:, i] - 1, minlength=5).tolist() for i in range(5)
        ],
        "episodes": len(episodes), "tasks": len(task_names),
        "base_frames": base_count, "labelled_frames": valid_count,
        "masked_frames": base_count - valid_count,
        "allowed_missing_suffix_per_episode": EXPECTED_TAIL,
        "class_counts": class_counts.tolist(), "fixed_count": fixed_count,
        "contact_note": (
            "ratio_fixed came from TTaekwan/robocasa_contact where available; "
            "the source card reports action-heuristic fallback for the remainder"
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
