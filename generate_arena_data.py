#!/usr/bin/env python3
"""
Production pipeline for preparing Arena experiment data.

This script runs the data generation pipeline for CJE experiments:
1. Extract prompts from ChatBot Arena dataset
2. Generate responses using different policies
3. Add judge scores (lightweight evaluation)
4. Add oracle labels (high-quality evaluation)
5. Compute log probabilities
6. Combine into a single dataset file

The prepared data can then be analyzed with analyze_dataset.py,
which handles calibration and CJE estimation.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiment_config import (
    POLICY_NAMES,
    BATCH_SIZES,
    DEFAULT_EXPERIMENT_PARAMS,
    validate_environment,
    print_experiment_config,
)


def run_command(
    cmd: str, check: bool = True, skip_if_exists: Optional[Path] = None
) -> subprocess.CompletedProcess:
    """Run a shell command and return the result.

    Args:
        cmd: Command to run
        check: Whether to raise error on non-zero exit
        skip_if_exists: Skip command if this file exists
    """
    # Check if we should skip
    if skip_if_exists and skip_if_exists.exists():
        print(f"⏭️  Skipping (output exists): {skip_if_exists}")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    # If command starts with "python ", prepend "poetry run "
    if cmd.strip().startswith("python "):
        cmd = "poetry run " + cmd

    print(f"\n📍 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.stdout.strip():
        # Show first 500 chars of output
        output = result.stdout[:500]
        if len(result.stdout) > 500:
            output += "..."
        print(f"Output: {output}")

    if check and result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")

    return result


def create_prompts(
    output_dir: Path, n_samples: int, seed: int, skip_existing: bool
) -> None:
    """Extract prompts from ChatBot Arena dataset.

    Args:
        output_dir: Directory to save prompts
        n_samples: Number of prompts to extract
        seed: Random seed for reproducibility
        skip_existing: Whether to skip if prompts file exists
    """
    prompts_file = output_dir / "prompts.jsonl"

    if skip_existing and prompts_file.exists():
        print(f"⏭️  Skipping prompt extraction (file exists): {prompts_file}")
        return

    print(f"Preparing {n_samples} prompts from ChatBot Arena dataset...")

    # Import the prepare function
    from data_generation.prepare_arena_data import prepare_arena_prompts

    # Generate prompts
    prompts = prepare_arena_prompts(
        n_samples=n_samples,
        output_file=str(prompts_file),
        seed=seed,
    )

    if len(prompts) < n_samples:
        print(f"⚠️  Warning: Only got {len(prompts)} prompts, expected {n_samples}")

    print(f"✅ Extracted {len(prompts)} Arena prompts: {prompts_file}")


def main() -> None:
    """Run the production data preparation pipeline."""
    parser = argparse.ArgumentParser(
        description="Prepare Arena experiment data for CJE analysis"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory for output files (default: data)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_EXPERIMENT_PARAMS["n_samples"],
        help=f"Number of samples to generate (default: {DEFAULT_EXPERIMENT_PARAMS['n_samples']})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_EXPERIMENT_PARAMS["max_tokens"],
        help=f"Maximum tokens per response (default: {DEFAULT_EXPERIMENT_PARAMS['max_tokens']})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_EXPERIMENT_PARAMS["seed"],
        help=f"Random seed for reproducibility (default: {DEFAULT_EXPERIMENT_PARAMS['seed']})",
    )
    parser.add_argument(
        "--skip-existing",
        "--resume",
        action="store_true",
        default=True,
        help="Skip steps where output files already exist (default: True)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing files (disables resume/skip-existing)",
    )
    parser.add_argument(
        "--no-resume",
        dest="skip_existing",
        action="store_false",
        help="Disable resume/skip-existing behavior",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZES["response_generation"],
        help=f"Save progress every N samples for resilience (default: {BATCH_SIZES['response_generation']}, set to 0 to disable)",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show experiment configuration and exit",
    )
    args = parser.parse_args()

    # Show config if requested
    if args.show_config:
        print_experiment_config()
        sys.exit(0)

    # If --force is set, disable skip_existing
    if args.force:
        args.skip_existing = False

    print("🚀 Starting Arena experiment data preparation...")
    print(f"   Samples: {args.n_samples}")
    print(f"   Max tokens: {args.max_tokens}")
    print(f"   Batch size: {args.batch_size if args.batch_size > 0 else 'disabled'}")
    print(
        f"   Mode: {'Force overwrite' if args.force else 'Resume (skip existing)' if args.skip_existing else 'Overwrite'}"
    )
    print(f"   Output directory: {args.data_dir}")

    # Setup data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)

    # Check for existing data and warn user
    existing_files = []
    important_files = [
        data_dir / "cje_dataset.jsonl",
        data_dir / "prompts.jsonl",
    ]

    # Check for response files
    for policy in POLICY_NAMES:
        important_files.append(data_dir / "responses" / f"{policy}_responses.jsonl")

    for file in important_files:
        if file.exists():
            existing_files.append(file)

    # Only warn if we're about to overwrite (not resuming and not forcing)
    if existing_files and not args.skip_existing and not args.force:
        print("\n⚠️  WARNING: The following files already exist:")
        for file in existing_files:
            print(f"   - {file}")
        print("\nThis operation will OVERWRITE these files!")
        print("Options:")
        print("  1. Continue and overwrite (type 'yes')")
        print("  2. Resume from existing files (default behavior, or use --resume)")
        print("  3. Force overwrite without asking (rerun with --force)")
        print("  4. Cancel (any other input)")

        response = (
            input("\nDo you want to continue and OVERWRITE? (yes/no): ").strip().lower()
        )
        if response != "yes":
            print("❌ Operation cancelled. No files were modified.")
            sys.exit(0)
        print("⚠️  Proceeding with overwrite...")

    # Create subdirectories
    (data_dir / "responses").mkdir(exist_ok=True)
    (data_dir / "logprobs").mkdir(exist_ok=True)

    # Step 1: Extract prompts from Arena dataset
    print("\n" + "=" * 60)
    print("Step 1: Extract prompts from ChatBot Arena")
    print("=" * 60)
    create_prompts(
        data_dir,
        n_samples=args.n_samples,
        seed=args.seed,
        skip_existing=args.skip_existing,
    )

    # Step 2: Generate responses
    print("\n" + "=" * 60)
    print("Step 2: Generate responses for each policy")
    print("=" * 60)

    # Check if all response files exist
    policies = POLICY_NAMES
    all_responses_exist = all(
        (data_dir / "responses" / f"{policy}_responses.jsonl").exists()
        for policy in policies
    )

    if args.skip_existing and all_responses_exist:
        print("⏭️  Skipping response generation (all files exist)")
    else:
        cmd = (
            f"python data_generation/generate_responses.py "
            f"--prompts {data_dir}/prompts.jsonl "
            f"--output-dir {data_dir}/responses "
            f"--max-responses {args.n_samples} "
            f"--max-tokens {args.max_tokens}"
        )
        if args.batch_size > 0:
            cmd += f" --batch-size {args.batch_size}"
        run_command(cmd)

    # Step 3: Add judge scores
    print("\n" + "=" * 60)
    print("Step 3: Add judge scores (lightweight evaluation)")
    print("=" * 60)

    for policy in policies:
        response_file = data_dir / "responses" / f"{policy}_responses.jsonl"

        # Use the new scoring script with resume capability
        cmd = (
            f"python data_generation/add_scores_with_resume.py "
            f"{response_file} "
            f"--type judge "
            f"--batch-size {BATCH_SIZES['judge_scoring']}"
        )
        if args.force:
            cmd += " --force"  # Only force rescore when explicitly requested

        run_command(cmd)

    # Step 4: Add oracle labels
    print("\n" + "=" * 60)
    print("Step 4: Add oracle labels (high-quality evaluation)")
    print("=" * 60)

    for policy in policies:
        response_file = data_dir / "responses" / f"{policy}_responses.jsonl"

        # Use the new scoring script with resume capability
        cmd = (
            f"python data_generation/add_scores_with_resume.py "
            f"{response_file} "
            f"--type oracle "
            f"--batch-size {BATCH_SIZES['oracle_scoring']}"
        )
        if args.force:
            cmd += " --force"  # Only force rescore when explicitly requested

        run_command(cmd)

    # Step 5: Compute log probabilities
    print("\n" + "=" * 60)
    print("Step 5: Compute log probabilities")
    print("=" * 60)

    # Check if all logprob files exist
    all_logprobs_exist = all(
        (data_dir / "logprobs" / f"{policy}_logprobs.jsonl").exists()
        for policy in policies
    )

    if args.skip_existing and all_logprobs_exist:
        print("⏭️  Skipping logprob computation (all files exist)")
    else:
        cmd = (
            f"python data_generation/compute_logprobs.py "
            f"--responses-dir {data_dir}/responses "
            f"--output-dir {data_dir}/logprobs"
        )
        if args.batch_size > 0:
            cmd += f" --batch-size {args.batch_size}"
        run_command(cmd)

    # Step 6: Prepare CJE dataset
    print("\n" + "=" * 60)
    print("Step 6: Combine data into CJE dataset")
    print("=" * 60)

    cje_dataset_file = data_dir / "cje_dataset.jsonl"

    run_command(
        f"python data_generation/prepare_cje_data.py "
        f"--responses-dir {data_dir}/responses "
        f"--logprobs-dir {data_dir}/logprobs "
        f"--output {cje_dataset_file}",
        skip_if_exists=cje_dataset_file if args.skip_existing else None,
    )

    print("\n✅ Data generation completed successfully!")
    print(f"📁 Output directory: {data_dir}")
    print(f"📊 Dataset file: {cje_dataset_file}")
    print("\nNext step:")
    print(f"  poetry run python analyze_dataset.py --data {cje_dataset_file}")


if __name__ == "__main__":
    # Validate environment
    if not validate_environment():
        sys.exit(1)

    main()
