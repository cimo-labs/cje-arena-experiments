#!/usr/bin/env python3
"""
Run Inspect MBPP evaluations for each policy.

This script runs the MBPP benchmark from inspect_evals for each policy configuration,
saving the evaluation logs for later extraction.

Usage:
    # Run all policies
    python run_inspect_mbpp.py

    # Run specific policy
    python run_inspect_mbpp.py --policy base

    # Dry run (show commands without executing)
    python run_inspect_mbpp.py --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from config import POLICIES, DATA_DIR, RAW_DIR, INSPECT_CONFIG


def run_inspect_eval(
    policy_name: str,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> bool:
    """Run Inspect MBPP evaluation for a single policy.

    Args:
        policy_name: Name of the policy to evaluate
        dry_run: If True, print command without executing
        limit: Optional limit on number of problems to evaluate

    Returns:
        True if successful, False otherwise
    """
    if policy_name not in POLICIES:
        print(f"Error: Unknown policy '{policy_name}'")
        print(f"Available policies: {list(POLICIES.keys())}")
        return False

    policy = POLICIES[policy_name]
    output_dir = RAW_DIR / policy_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "inspect", "eval", INSPECT_CONFIG["task"],
        "--model", policy["model"],
        "--log-dir", str(output_dir),
        "--log-format", INSPECT_CONFIG["log_format"],
        "--max-connections", str(INSPECT_CONFIG["max_connections"]),
    ]

    # Add system prompt via model args
    # Inspect uses --model-args for additional model configuration
    system_prompt = policy["system_prompt"].replace('"', '\\"')
    cmd.extend(["--system-message", policy["system_prompt"]])

    # Add temperature
    if "temperature" in policy:
        cmd.extend(["--temperature", str(policy["temperature"])])

    # Add limit if specified
    if limit:
        cmd.extend(["--limit", str(limit)])

    # Add timeout
    if "timeout" in INSPECT_CONFIG:
        cmd.extend(["--timeout", str(INSPECT_CONFIG["timeout"])])

    # Use local sandbox (avoids Docker version issues)
    cmd.extend(["--sandbox", "local"])

    print(f"\n{'='*60}")
    print(f"Running MBPP evaluation for policy: {policy_name}")
    print(f"{'='*60}")
    print(f"Model: {policy['model']}")
    print(f"Output: {output_dir}")
    print(f"Description: {policy.get('description', 'N/A')}")

    if dry_run:
        print(f"\n[DRY RUN] Would execute:")
        print(f"  {' '.join(cmd)}")
        return True

    print(f"\nExecuting: {' '.join(cmd[:6])}...")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
        )
        print(f"\n✓ Successfully completed evaluation for {policy_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to run evaluation for {policy_name}: {e}")
        return False
    except FileNotFoundError:
        print("\n✗ Error: 'inspect' command not found.")
        print("  Please install inspect-ai and inspect-evals:")
        print("  pip install inspect-ai inspect-evals")
        return False


def run_all_policies(
    dry_run: bool = False,
    limit: Optional[int] = None,
    policies: Optional[List[str]] = None,
) -> dict:
    """Run MBPP evaluations for all (or specified) policies.

    Args:
        dry_run: If True, print commands without executing
        limit: Optional limit on number of problems
        policies: Optional list of specific policies to run

    Returns:
        Dictionary with results per policy
    """
    results = {}
    policy_list = policies if policies else list(POLICIES.keys())

    print(f"\n{'#'*60}")
    print(f"# MBPP Evaluation Run")
    print(f"# Policies: {', '.join(policy_list)}")
    print(f"# Output: {RAW_DIR}")
    print(f"{'#'*60}")

    for policy_name in policy_list:
        success = run_inspect_eval(policy_name, dry_run=dry_run, limit=limit)
        results[policy_name] = "success" if success else "failed"

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for policy, status in results.items():
        icon = "✓" if status == "success" else "✗"
        print(f"  {icon} {policy}: {status}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Inspect MBPP evaluations for CJE policies"
    )
    parser.add_argument(
        "--policy", "-p",
        type=str,
        help="Specific policy to run (default: all policies)",
    )
    parser.add_argument(
        "--policies",
        type=str,
        nargs="+",
        help="List of policies to run",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="Limit number of problems to evaluate (for testing)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available policies and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("Available policies:")
        for name, policy in POLICIES.items():
            print(f"  {name}: {policy.get('description', 'N/A')}")
        return 0

    # Determine which policies to run
    if args.policy:
        policies = [args.policy]
    elif args.policies:
        policies = args.policies
    else:
        policies = None  # All policies

    results = run_all_policies(
        dry_run=args.dry_run,
        limit=args.limit,
        policies=policies,
    )

    # Return non-zero if any failed
    if "failed" in results.values():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
