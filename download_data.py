#!/usr/bin/env python3
"""
Download CJE Arena experiment data from HuggingFace.

This script downloads the required data files for running the CJE arena experiments.
Data is hosted at: https://huggingface.co/datasets/cimo-labs/cje-arena-data

Usage:
    python download_data.py              # Download all data
    python download_data.py --no-results # Skip large results file (1.3GB)
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# Dataset repository on HuggingFace
HF_REPO = "cimo-labs/cje-arena-data"

# File manifest with expected checksums (SHA256)
FILE_MANIFEST = {
    # Core dataset
    "cje_dataset.jsonl": {
        "path": "data/cje_dataset.jsonl",
        "required": True,
        "description": "Main dataset with prompts, responses, and logprobs",
    },
    "prompts.jsonl": {
        "path": "data/prompts.jsonl",
        "required": True,
        "description": "Original Arena prompts",
    },
    # Fresh draws for DR estimators
    "responses/base_responses.jsonl": {
        "path": "data/responses/base_responses.jsonl",
        "required": False,
        "description": "Fresh draws from base policy",
    },
    "responses/clone_responses.jsonl": {
        "path": "data/responses/clone_responses.jsonl",
        "required": False,
        "description": "Fresh draws from clone policy",
    },
    "responses/parallel_universe_prompt_responses.jsonl": {
        "path": "data/responses/parallel_universe_prompt_responses.jsonl",
        "required": False,
        "description": "Fresh draws from parallel universe policy",
    },
    "responses/premium_responses.jsonl": {
        "path": "data/responses/premium_responses.jsonl",
        "required": False,
        "description": "Fresh draws from premium policy",
    },
    "responses/unhelpful_responses.jsonl": {
        "path": "data/responses/unhelpful_responses.jsonl",
        "required": False,
        "description": "Fresh draws from unhelpful policy",
    },
    # Pre-computed results (optional, large)
    "results/all_experiments.jsonl": {
        "path": "ablations/results/all_experiments.jsonl",
        "required": False,
        "description": "Pre-computed ablation results (1.3GB)",
        "large": True,
    },
}


def check_huggingface_hub():
    """Check if huggingface_hub is installed."""
    if not HF_AVAILABLE:
        print("ERROR: huggingface_hub not installed.")
        print("Install with: pip install huggingface_hub")
        return False
    return True


def download_file(repo_file: str, local_path: Path, repo_id: str = HF_REPO) -> bool:
    """Download a single file from HuggingFace."""
    try:
        # Create parent directories
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download file
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=repo_file,
            repo_type="dataset",
            local_dir=str(local_path.parent.parent),
            local_dir_use_symlinks=False,
        )

        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def download_all(include_results: bool = True, force: bool = False):
    """Download all required data files."""
    if not check_huggingface_hub():
        return False

    print(f"Downloading data from HuggingFace: {HF_REPO}")
    print()

    base_dir = Path(__file__).parent
    success_count = 0
    skip_count = 0
    fail_count = 0

    for repo_file, info in FILE_MANIFEST.items():
        local_path = base_dir / info["path"]

        # Skip large files if not requested
        if info.get("large") and not include_results:
            print(f"  SKIP: {repo_file} (use --include-results to download)")
            skip_count += 1
            continue

        # Check if file exists
        if local_path.exists() and not force:
            print(f"  EXISTS: {info['path']}")
            success_count += 1
            continue

        # Download
        print(f"  Downloading: {repo_file}...", end=" ", flush=True)
        if download_file(repo_file, local_path):
            print("OK")
            success_count += 1
        else:
            fail_count += 1

    print()
    print(f"Downloaded: {success_count}, Skipped: {skip_count}, Failed: {fail_count}")

    return fail_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="Download CJE Arena experiment data from HuggingFace"
    )
    parser.add_argument(
        "--no-results",
        action="store_true",
        help="Skip downloading large results file (1.3GB)",
    )
    parser.add_argument(
        "--include-results",
        action="store_true",
        help="Include large results file (1.3GB)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files exist",
    )

    args = parser.parse_args()

    # Default: don't download results unless explicitly requested
    include_results = args.include_results and not args.no_results

    success = download_all(include_results=include_results, force=args.force)

    if success:
        print()
        print("Data download complete!")
        print("Run 'python verify_setup.py' to verify the setup.")
    else:
        print()
        print("Some downloads failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
