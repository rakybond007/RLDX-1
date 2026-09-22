#!/usr/bin/env python3
"""Download the immutable contact-aware ratio-label release from Hugging Face."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "prehj/robocasa-ratio-labels-contact"
REVISION = "c9d9fbd76a863cdac76afe8f09269c19e48bc1bf"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--token-file", default="")
    args = parser.parse_args()
    token = None
    if args.token_file:
        token = Path(args.token_file).read_text().strip()
    result = snapshot_download(
        repo_id=REPO_ID, repo_type="dataset", revision=REVISION,
        local_dir=args.out, token=token,
    )
    print(f"downloaded {REPO_ID}@{REVISION} -> {result}")


if __name__ == "__main__":
    main()
