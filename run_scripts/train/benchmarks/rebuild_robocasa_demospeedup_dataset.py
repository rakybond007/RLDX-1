"""Verify the released RoboCasa entropy kit and rebuild its full training dataset."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    source_info = json.loads((args.source / "meta/info.json").read_text())
    reference_info = json.loads((args.kit / "reference/source_info.json").read_text())
    # The local LeRobot copy adds an image_path convenience field. It does not
    # change the verified parquet/video paths or any feature used for training.
    assert {key: source_info[key] for key in reference_info} == reference_info
    manifest = json.loads((args.kit / "reference/manifest.json").read_text())
    assert len(manifest) == source_info["total_episodes"] == 7200
    assert sum(row["frames"] for row in manifest) == source_info["total_frames"]
    for episode_index, row in enumerate(manifest):
        assert row["episode_index"] == episode_index
        relative = source_info["data_path"].format(
            episode_chunk=episode_index // source_info["chunks_size"], episode_index=episode_index
        )
        assert row["path"] == relative
        assert digest(args.source / relative) == row["source_sha256"], relative
        assert digest(args.kit / "entropy" / f"episode_{episode_index:06d}.npz") == row["entropy_sha256"]
    print("SOURCE_AND_ENTROPY_VERIFIED", flush=True)
    if not args.verify_only:
        subprocess.run([
            sys.executable, str(args.kit / "code/package_dataset.py"),
            "--source", str(args.source), "--entropy", str(args.kit / "entropy"),
            "--out", str(args.out),
        ], check=True)


if __name__ == "__main__":
    main()
