"""Download FLUX.1-schnell activation/weight dump from Hugging Face."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

PUBLIC_REPO = "samuelt0207/cuda-challenge-flux-dump"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=PUBLIC_REPO,
                        help=f"HF dataset repo to pull from (default: {PUBLIC_REPO})")
    parser.add_argument("--out-dir", type=Path, default=Path("flux_dump"),
                        help="Local directory to populate (default: flux_dump)")
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True, parents=True)
    path = snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=str(args.out_dir),
        allow_patterns=["*.pt"],
    )
    print(f"Downloaded to {path}")


if __name__ == "__main__":
    main()
