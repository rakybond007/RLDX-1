"""Build a small, source-verified RoboCasa DemoSpeedup fixture for training smoke tests."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    args = parser.parse_args()
    assert 1 <= args.episodes <= 7200
    assert not args.out.exists(), f"Output already exists: {args.out}"
    sys.path.insert(0, str(args.kit / "code"))
    from method import make_targets
    from package_dataset import validate_targets

    source_info = json.loads((args.source / "meta/info.json").read_text())
    reference_info = json.loads((args.kit / "reference/source_info.json").read_text())
    assert {key: source_info[key] for key in reference_info} == reference_info
    manifest = json.loads((args.kit / "reference/manifest.json").read_text())
    episodes = [json.loads(line) for line in (args.source / "meta/episodes.jsonl").read_text().splitlines()]
    args.out.mkdir(parents=True)
    meta = args.out / "meta"
    meta.mkdir()
    for filename in ("stats.json", "relative_stats.json", "modality.json", "tasks.jsonl"):
        source = args.source / "meta" / filename
        if source.exists():
            (meta / filename).write_bytes(source.read_bytes())
    (meta / "episodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in episodes[: args.episodes]))
    info = source_info.copy()
    info["total_episodes"] = args.episodes
    info["total_frames"] = sum(x["length"] for x in episodes[: args.episodes])
    info["total_chunks"] = (args.episodes + info["chunks_size"] - 1) // info["chunks_size"]
    info["splits"] = {"train": f"0:{args.episodes}"}
    info["features"] = dict(info["features"])
    info["features"]["speedup.actions"] = {"dtype": "float32", "shape": [96], "names": None}
    info["features"]["speedup.valid"] = {"dtype": "bool", "shape": [8], "names": None}
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    (args.out / "videos").symlink_to((args.source / "videos").resolve(), target_is_directory=True)

    values = []
    for episode_index in range(args.episodes):
        record = manifest[episode_index]
        assert record["episode_index"] == episode_index
        relative = source_info["data_path"].format(
            episode_chunk=episode_index // source_info["chunks_size"], episode_index=episode_index
        )
        assert record["path"] == relative
        source_path = args.source / relative
        assert sha256(source_path) == record["source_sha256"]
        entropy_path = args.kit / "entropy" / f"episode_{episode_index:06d}.npz"
        assert sha256(entropy_path) == record["entropy_sha256"]
        table = pq.read_table(source_path)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        with np.load(entropy_path) as entropy:
            assert int(entropy["episode_index"]) == episode_index
            precision = entropy["precision"].astype(bool)
        targets, valid, starts, ends, rates = make_targets(actions, precision, horizon=8)
        validate_targets(actions, precision, targets, valid, starts, ends, rates)
        table = table.append_column(
            "speedup.actions", pa.array(targets.reshape(len(actions), 96).tolist(), type=pa.list_(pa.float32(), 96))
        ).append_column("speedup.valid", pa.array(valid.tolist(), type=pa.list_(pa.bool_(), 8)))
        output_path = args.out / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path, compression="zstd")
        values.append(targets[valid])
        print(f"SMOKE_EPISODE {episode_index} frames={len(actions)} valid={int(valid.sum())}", flush=True)

    flat = np.concatenate(values)
    stats = {name: value.tolist() for name, value in {
        "mean": flat.mean(0, dtype=np.float64), "std": flat.std(0, dtype=np.float64),
        "min": flat.min(0), "max": flat.max(0),
        "q01": np.quantile(flat, 0.01, axis=0), "q99": np.quantile(flat, 0.99, axis=0),
    }.items()}
    stats["count"] = len(flat)
    (meta / "speedup_action_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"SMOKE_DATASET_READY episodes={args.episodes} frames={info['total_frames']} targets={len(flat)}", flush=True)


if __name__ == "__main__":
    main()
