#!/usr/bin/env python3
"""
Generate additional passes for log probability computation to study API non-determinism.

This script orchestrates multiple runs of compute_logprobs.py with different pass numbers,
allowing us to:
1. Document variance in API responses
2. Identify deterministic vs non-deterministic failures
3. Improve data quality through aggregation (in analysis phase)

Usage:
    # Generate passes 2-5 for all policies
    python generate_additional_passes.py --n-passes 5

    # Generate specific passes
    python generate_additional_passes.py --start-pass 3 --end-pass 5

    # Run in parallel (faster but more API load)
    python generate_additional_passes.py --n-passes 5 --parallel
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import concurrent.futures
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from experiment_config import POLICY_NAMES, BATCH_SIZES


def check_existing_passes(logprobs_dir: Path) -> Dict[str, List[int]]:
    """Check which passes already exist for each policy.

    Returns:
        Dictionary mapping policy name to list of existing pass numbers
    """
    existing: Dict[str, List[int]] = defaultdict(list)

    if not logprobs_dir.exists():
        return existing

    # Check for original files (pass 1)
    for policy in POLICY_NAMES:
        original = logprobs_dir / f"{policy}_logprobs.jsonl"
        if original.exists():
            existing[policy].append(1)

    # Check for additional passes
    for logprob_file in logprobs_dir.glob("*_logprobs_pass*.jsonl"):
        # Extract policy and pass number from filename
        parts = logprob_file.stem.split("_logprobs_pass")
        if len(parts) == 2:
            policy = parts[0]
            try:
                pass_num = int(parts[1])
                existing[policy].append(pass_num)
            except ValueError:
                print(f"Warning: Could not parse pass number from {logprob_file.name}")

    return existing


def run_single_pass(
    policy: str,
    pass_number: int,
    responses_dir: Path,
    output_dir: Path,
    batch_size: int,
) -> subprocess.CompletedProcess:
    """Run compute_logprobs.py for a single policy and pass."""

    cmd = [
        "python",
        "data_generation/compute_logprobs.py",
        "--responses-dir",
        str(responses_dir),
        "--output-dir",
        str(output_dir),
        "--policies",
        policy,
        "--pass-number",
        str(pass_number),
        "--batch-size",
        str(batch_size),
    ]

    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True)


def run_sequential_passes(
    passes_to_run: List[Tuple[str, int]],
    responses_dir: Path,
    output_dir: Path,
    batch_size: int,
) -> None:
    """Run passes sequentially."""

    total = len(passes_to_run)
    completed = 0
    failed = []

    for policy, pass_num in passes_to_run:
        print(f"\n[{completed + 1}/{total}] Processing {policy} pass {pass_num}...")

        result = run_single_pass(
            policy, pass_num, responses_dir, output_dir, batch_size
        )

        if result.returncode == 0:
            completed += 1
            print(f"✓ Completed {policy} pass {pass_num}")
        else:
            failed.append((policy, pass_num))
            print(f"✗ Failed {policy} pass {pass_num}")
            print(f"  Error: {result.stderr[:500]}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Completed: {completed}/{total}")
    if failed:
        print(f"Failed: {len(failed)}")
        for policy, pass_num in failed:
            print(f"  - {policy} pass {pass_num}")


def run_parallel_passes(
    passes_to_run: List[Tuple[str, int]],
    responses_dir: Path,
    output_dir: Path,
    batch_size: int,
    max_workers: int = 4,
) -> None:
    """Run passes in parallel using process pool."""

    total = len(passes_to_run)
    completed = 0
    failed = []

    print(f"Running {total} passes in parallel (max {max_workers} workers)...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_pass = {}
        for policy, pass_num in passes_to_run:
            future = executor.submit(
                run_single_pass, policy, pass_num, responses_dir, output_dir, batch_size
            )
            future_to_pass[future] = (policy, pass_num)

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_pass):
            policy, pass_num = future_to_pass[future]
            try:
                result = future.result()
                if result.returncode == 0:
                    completed += 1
                    print(f"✓ [{completed}/{total}] Completed {policy} pass {pass_num}")
                else:
                    failed.append((policy, pass_num))
                    print(f"✗ Failed {policy} pass {pass_num}")
            except Exception as e:
                failed.append((policy, pass_num))
                print(f"✗ Exception for {policy} pass {pass_num}: {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Completed: {completed}/{total}")
    if failed:
        print(f"Failed: {len(failed)}")
        for policy, pass_num in failed:
            print(f"  - {policy} pass {pass_num}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate additional passes for log probability computation"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory containing responses/ and logprobs/ subdirectories",
    )
    parser.add_argument(
        "--n-passes",
        type=int,
        default=5,
        help="Total number of passes to have (including original)",
    )
    parser.add_argument(
        "--start-pass",
        type=int,
        help="Starting pass number (default: 2 if original exists, else 1)",
    )
    parser.add_argument(
        "--end-pass", type=int, help="Ending pass number (default: n-passes)"
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=POLICY_NAMES,
        help="Specific policies to process (default: all)",
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Run passes in parallel"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum parallel workers (default: 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZES["logprob_computation"],
        help=f"Batch size for saving progress (default: {BATCH_SIZES['logprob_computation']})",
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate even if pass already exists"
    )

    args = parser.parse_args()

    # Setup directories
    responses_dir = args.data_dir / "responses"
    logprobs_dir = args.data_dir / "logprobs"

    if not responses_dir.exists():
        print(f"Error: Responses directory not found: {responses_dir}")
        sys.exit(1)

    logprobs_dir.mkdir(parents=True, exist_ok=True)

    # Check what already exists
    existing = check_existing_passes(logprobs_dir)

    print("Existing passes:")
    for policy in args.policies:
        if policy in existing:
            print(f"  {policy}: passes {sorted(existing[policy])}")
        else:
            print(f"  {policy}: none")

    # Determine which passes to run
    passes_to_run = []

    # Determine pass range
    start_pass = (
        args.start_pass if args.start_pass else 2
    )  # Default to 2 (skip original)
    end_pass = args.end_pass if args.end_pass else args.n_passes

    for policy in args.policies:
        for pass_num in range(start_pass, end_pass + 1):
            # Skip if already exists (unless force)
            if not args.force and pass_num in existing.get(policy, []):
                print(f"Skipping {policy} pass {pass_num} (already exists)")
                continue

            passes_to_run.append((policy, pass_num))

    if not passes_to_run:
        print("\nNo passes to generate. All requested passes already exist.")
        print("Use --force to regenerate existing passes.")
        return

    print(f"\nWill generate {len(passes_to_run)} passes:")
    for policy, pass_num in passes_to_run[:5]:
        print(f"  - {policy} pass {pass_num}")
    if len(passes_to_run) > 5:
        print(f"  ... and {len(passes_to_run) - 5} more")

    # Run the passes
    if args.parallel:
        run_parallel_passes(
            passes_to_run,
            responses_dir,
            logprobs_dir,
            args.batch_size,
            args.max_workers,
        )
    else:
        run_sequential_passes(
            passes_to_run, responses_dir, logprobs_dir, args.batch_size
        )

    print("\n✅ Additional pass generation complete!")
    print("Next step: Run analysis to study API non-determinism")
    print("  python data_generation/analyze_nondeterminism.py --data-dir data")


if __name__ == "__main__":
    main()
