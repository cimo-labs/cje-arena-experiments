#!/usr/bin/env python3
"""
Format MBPP responses into CJE dataset format.

Creates the main CJE dataset (base policy) and validates that all
target policy responses are properly formatted for ablation experiments.

Usage:
    # Format all data
    python format_cje_dataset.py

    # Validate only (no changes)
    python format_cje_dataset.py --validate

    # Specify base policy
    python format_cje_dataset.py --base-policy enhanced
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict
import numpy as np

from config import POLICIES, DATA_DIR, RESPONSES_DIR


def load_policy_responses(policy_name: str) -> List[Dict[str, Any]]:
    """Load responses for a policy.

    Args:
        policy_name: Name of the policy

    Returns:
        List of response records
    """
    response_file = RESPONSES_DIR / f"{policy_name}_responses.jsonl"

    if not response_file.exists():
        raise FileNotFoundError(f"Response file not found: {response_file}")

    records = []
    with open(response_file, "r") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def validate_responses(
    records: List[Dict[str, Any]],
    policy_name: str,
    require_judge_score: bool = True,
    require_oracle_label: bool = True,
) -> Dict[str, Any]:
    """Validate response records.

    Args:
        records: List of response records
        policy_name: Name of the policy (for error messages)
        require_judge_score: Whether judge_score is required
        require_oracle_label: Whether oracle_label is required

    Returns:
        Validation statistics
    """
    stats = {
        "total": len(records),
        "valid": 0,
        "missing_prompt_id": 0,
        "missing_prompt": 0,
        "missing_response": 0,
        "missing_judge_score": 0,
        "missing_oracle_label": 0,
        "errors": [],
    }

    prompt_ids: Set[str] = set()

    for i, record in enumerate(records):
        is_valid = True

        # Check prompt_id
        prompt_id = record.get("prompt_id")
        if not prompt_id:
            stats["missing_prompt_id"] += 1
            is_valid = False
        else:
            if prompt_id in prompt_ids:
                stats["errors"].append(f"Duplicate prompt_id: {prompt_id}")
            prompt_ids.add(prompt_id)

        # Check prompt
        if not record.get("prompt"):
            stats["missing_prompt"] += 1
            is_valid = False

        # Check response
        if not record.get("response"):
            stats["missing_response"] += 1
            is_valid = False

        # Check judge_score in metadata
        metadata = record.get("metadata", {})
        if require_judge_score and metadata.get("judge_score") is None:
            stats["missing_judge_score"] += 1
            is_valid = False

        # Check oracle_label in metadata
        if require_oracle_label and metadata.get("oracle_label") is None:
            stats["missing_oracle_label"] += 1
            is_valid = False

        if is_valid:
            stats["valid"] += 1

    return stats


def check_prompt_alignment(
    base_records: List[Dict[str, Any]],
    target_records_dict: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Check that all policies have the same prompt_ids.

    Args:
        base_records: Base policy records
        target_records_dict: Dict mapping policy name to records

    Returns:
        Alignment statistics
    """
    base_prompt_ids = {r.get("prompt_id") for r in base_records if r.get("prompt_id")}

    alignment = {
        "base_count": len(base_prompt_ids),
        "all_aligned": True,
        "policies": {},
    }

    for policy_name, records in target_records_dict.items():
        policy_prompt_ids = {r.get("prompt_id") for r in records if r.get("prompt_id")}

        missing_from_target = base_prompt_ids - policy_prompt_ids
        extra_in_target = policy_prompt_ids - base_prompt_ids

        alignment["policies"][policy_name] = {
            "count": len(policy_prompt_ids),
            "missing": len(missing_from_target),
            "extra": len(extra_in_target),
        }

        if missing_from_target or extra_in_target:
            alignment["all_aligned"] = False

    return alignment


def format_cje_dataset(
    base_policy: str = "base",
    output_path: Optional[Path] = None,
    validate_only: bool = False,
) -> Dict[str, Any]:
    """Format MBPP responses into CJE dataset.

    The CJE dataset format:
    - prompt: The problem description
    - response: Base policy's code solution
    - metadata: prompt_id, judge_score, oracle_label, response_length

    Target policy responses are stored separately in responses/ directory.

    Args:
        base_policy: Name of the base (logging) policy
        output_path: Path for output CJE dataset
        validate_only: If True, only validate without creating output

    Returns:
        Statistics dictionary
    """
    output_path = output_path or (DATA_DIR / "mbpp_dataset.jsonl")

    print(f"\n{'='*60}")
    print(f"Formatting CJE Dataset")
    print(f"{'='*60}")
    print(f"Base policy: {base_policy}")
    print(f"Output: {output_path}")

    # Load base policy responses
    print(f"\nLoading base policy responses...")
    try:
        base_records = load_policy_responses(base_policy)
        print(f"  Loaded {len(base_records)} records")
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        return {"error": str(e)}

    # Validate base policy
    print(f"\nValidating base policy...")
    base_stats = validate_responses(base_records, base_policy)
    print(f"  Valid: {base_stats['valid']}/{base_stats['total']}")
    if base_stats["missing_judge_score"] > 0:
        print(f"  Warning: {base_stats['missing_judge_score']} missing judge scores")
    if base_stats["missing_oracle_label"] > 0:
        print(f"  Warning: {base_stats['missing_oracle_label']} missing oracle labels")

    # Load and validate target policies
    target_policies = [p for p in POLICIES.keys() if p != base_policy]
    target_records_dict = {}

    print(f"\nValidating target policies...")
    for policy_name in target_policies:
        try:
            records = load_policy_responses(policy_name)
            target_records_dict[policy_name] = records

            stats = validate_responses(records, policy_name)
            status = "✓" if stats["valid"] == stats["total"] else "⚠"
            print(f"  {status} {policy_name}: {stats['valid']}/{stats['total']} valid")

            if stats["missing_judge_score"] > 0:
                print(f"     - {stats['missing_judge_score']} missing judge scores")

        except FileNotFoundError:
            print(f"  ✗ {policy_name}: file not found")
            target_records_dict[policy_name] = []

    # Check prompt alignment
    print(f"\nChecking prompt alignment...")
    alignment = check_prompt_alignment(base_records, target_records_dict)
    if alignment["all_aligned"]:
        print(f"  ✓ All policies have {alignment['base_count']} aligned prompts")
    else:
        print(f"  ⚠ Prompt alignment issues detected:")
        for policy_name, info in alignment["policies"].items():
            if info["missing"] > 0 or info["extra"] > 0:
                print(f"    {policy_name}: {info['missing']} missing, {info['extra']} extra")

    if validate_only:
        print(f"\n[Validate only - no output created]")
        return {
            "base_stats": base_stats,
            "alignment": alignment,
            "output": None,
        }

    # Create CJE dataset from base policy
    print(f"\nCreating CJE dataset...")
    cje_records = []

    for record in base_records:
        metadata = record.get("metadata", {})

        # Build CJE record
        # For Direct mode, we add target_policy_logprobs with None values
        # to tell the CJE loader which target policies exist
        target_policy_logprobs = {p: None for p in target_policies}

        cje_record = {
            "prompt": record.get("prompt", ""),
            "response": record.get("response", ""),
            "base_policy_logprob": None,  # Not used in Direct mode
            "target_policy_logprobs": target_policy_logprobs,
            "metadata": {
                "prompt_id": record.get("prompt_id") or metadata.get("prompt_id"),
                "judge_score": metadata.get("judge_score"),
                "oracle_label": metadata.get("oracle_label"),
                "response_length": len(record.get("response", "").split()),
            },
        }

        # Include test_passed if available
        if "test_passed" in metadata:
            cje_record["metadata"]["test_passed"] = metadata["test_passed"]

        cje_records.append(cje_record)

    # Save CJE dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in cje_records:
            f.write(json.dumps(record) + "\n")

    print(f"  Created {len(cje_records)} records")
    print(f"  Saved to: {output_path}")

    # Statistics
    judge_scores = [
        r["metadata"]["judge_score"]
        for r in cje_records
        if r["metadata"].get("judge_score") is not None
    ]
    oracle_labels = [
        r["metadata"]["oracle_label"]
        for r in cje_records
        if r["metadata"].get("oracle_label") is not None
    ]

    print(f"\nDataset Statistics:")
    print(f"  Total records: {len(cje_records)}")

    if judge_scores:
        print(f"  Judge scores: mean={np.mean(judge_scores):.3f}, std={np.std(judge_scores):.3f}")

    if oracle_labels:
        pass_rate = np.mean(oracle_labels)
        print(f"  Oracle pass rate: {pass_rate:.1%} ({sum(oracle_labels)}/{len(oracle_labels)})")

    return {
        "base_stats": base_stats,
        "alignment": alignment,
        "output": str(output_path),
        "n_records": len(cje_records),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Format MBPP responses into CJE dataset"
    )
    parser.add_argument(
        "--base-policy", "-b",
        type=str,
        default="base",
        help="Base (logging) policy name (default: base)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help=f"Output path (default: {DATA_DIR / 'mbpp_dataset.jsonl'})",
    )
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Validate only, don't create output",
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

    result = format_cje_dataset(
        base_policy=args.base_policy,
        output_path=args.output,
        validate_only=args.validate,
    )

    if "error" in result:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
