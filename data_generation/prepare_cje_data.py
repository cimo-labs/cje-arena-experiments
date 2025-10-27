#!/usr/bin/env python3
"""
Combine responses and log probabilities into a single dataset file.

This script performs data organization only (no modeling or calibration):
- Uses BASE policy responses for all samples
- Includes log probabilities under all policy models
- Preserves judge scores and oracle labels in metadata
- Creates a single JSONL file ready for analysis

Note: Calibration of judge scores to rewards happens during analysis,
not during data preparation.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from collections import defaultdict

import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))  # Add arena_10k_simplified to path


def load_policy_passes(
    logprobs_dir: Path, policy: str
) -> Tuple[Dict[str, List[Optional[float]]], int]:
    """Load all passes for a policy and return passes by prompt_id.

    Returns:
        Tuple of (passes_by_prompt, num_passes_found)
        where passes_by_prompt maps prompt_id -> list of logprob values
    """
    passes_by_prompt = defaultdict(list)
    passes_found = 0

    # Load pass 1 (original file)
    pass1_file = logprobs_dir / f"{policy}_logprobs.jsonl"
    if pass1_file.exists():
        passes_found = 1
        prompt_logprobs = {}
        with open(pass1_file, "r") as f:
            for line in f:
                data = json.loads(line)
                prompt_id = data["prompt_id"]
                prompt_logprobs[prompt_id] = data.get("logprob")

        # Add to passes list
        for prompt_id in prompt_logprobs:
            passes_by_prompt[prompt_id].append(prompt_logprobs[prompt_id])

    # Load additional passes (pass 2, 3, ...)
    pass_num = 2
    while True:
        pass_file = logprobs_dir / f"{policy}_logprobs_pass{pass_num}.jsonl"
        if not pass_file.exists():
            break

        passes_found = pass_num
        prompt_logprobs = {}
        with open(pass_file, "r") as f:
            for line in f:
                data = json.loads(line)
                prompt_id = data["prompt_id"]
                prompt_logprobs[prompt_id] = data.get("logprob")

        # Add to passes list (aligning by prompt_id)
        for prompt_id in passes_by_prompt:
            if prompt_id in prompt_logprobs:
                passes_by_prompt[prompt_id].append(prompt_logprobs[prompt_id])
            else:
                # Missing from this pass
                passes_by_prompt[prompt_id].append(None)

        pass_num += 1

    return dict(passes_by_prompt), passes_found


def aggregate_passes(
    passes: List[Optional[float]], method: str = "mean"
) -> Optional[float]:
    """Aggregate multiple logprob passes using specified method.

    Args:
        passes: List of logprob values (may contain None)
        method: "mean" or "median"

    Returns:
        Aggregated value or None if no valid passes
    """
    # Filter out None values and positive values (invalid)
    valid_passes = [p for p in passes if p is not None and p <= 0]

    if not valid_passes:
        return None

    if len(valid_passes) == 1:
        return valid_passes[0]

    if method == "median":
        return float(np.median(valid_passes))
    else:  # mean
        return float(np.mean(valid_passes))


def prepare_cje_dataset(
    logprobs_dir: str,
    responses_dir: str,
    output_file: Optional[str],
    base_policy: str = "base",
    aggregation: str = "mean",  # "mean" or "median"
) -> List[Dict]:
    """Combine BASE policy responses with log probs from all policies.

    Creates records in the format expected by the CJE data model:
    - prompt: The input prompt
    - response: The BASE policy's response
    - base_policy_logprob: Log P(response | prompt) under base policy
    - target_policy_logprobs: Dict of log P(response | prompt) under each policy
    - metadata: Additional fields including judge_score for calibration
    """

    print("Preparing CJE dataset...")
    print(f"Aggregation method: {aggregation}")

    # First, load base responses to get judge/oracle scores
    responses_by_prompt: Dict[str, Dict[str, Any]] = {}
    base_responses_file = Path(responses_dir) / f"{base_policy}_responses.jsonl"

    print(f"Loading base responses from {base_responses_file}...")
    with open(base_responses_file, "r") as f:
        for line in f:
            data = json.loads(line)
            prompt_id = data.get("prompt_id")
            if prompt_id:
                responses_by_prompt[prompt_id] = data

    print(f"Loaded {len(responses_by_prompt)} base responses with evaluation scores")

    # Check if multiple passes exist
    logprobs_path = Path(logprobs_dir)
    has_multiple_passes = len(list(logprobs_path.glob("*_logprobs_pass*.jsonl"))) > 0

    if has_multiple_passes:
        print("\n📊 Multiple passes detected - will aggregate using", aggregation)

    # Get list of all policies from files
    policies: Set[str] = set()
    for file in logprobs_path.glob("*_logprobs.jsonl"):
        policy = file.stem.replace("_logprobs", "")
        policies.add(policy)

    # Load and aggregate log probabilities for each policy
    logprobs_by_prompt: Dict[str, Dict[str, Any]] = defaultdict(dict)

    for policy in sorted(policies):
        passes_by_prompt, num_passes = load_policy_passes(logprobs_path, policy)

        if num_passes > 1:
            print(f"Loading {policy} log probabilities... ({num_passes} passes found)")
        else:
            print(f"Loading {policy} log probabilities...")

        # Get sample data for prompt/response (from first pass)
        sample_data = {}
        pass1_file = logprobs_path / f"{policy}_logprobs.jsonl"
        if pass1_file.exists():
            with open(pass1_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    sample_data[data["prompt_id"]] = {
                        "prompt": data["prompt"],
                        "response": data["response"],
                    }

        # Process each prompt
        for prompt_id, passes in passes_by_prompt.items():
            # Store prompt/response if not already stored
            if (
                "prompt" not in logprobs_by_prompt[prompt_id]
                and prompt_id in sample_data
            ):
                logprobs_by_prompt[prompt_id]["prompt"] = sample_data[prompt_id][
                    "prompt"
                ]
                logprobs_by_prompt[prompt_id]["response"] = sample_data[prompt_id][
                    "response"
                ]
                logprobs_by_prompt[prompt_id]["prompt_id"] = prompt_id

            # Aggregate the passes
            aggregated_value = aggregate_passes(passes, method=aggregation)

            # Store aggregated log prob
            if policy == base_policy:
                logprobs_by_prompt[prompt_id]["base_policy_logprob"] = aggregated_value
            else:
                if "target_policy_logprobs" not in logprobs_by_prompt[prompt_id]:
                    logprobs_by_prompt[prompt_id]["target_policy_logprobs"] = {}
                logprobs_by_prompt[prompt_id]["target_policy_logprobs"][
                    policy
                ] = aggregated_value

    print(f"Found {len(policies)} policies: {sorted(policies)}")

    # Create CJE format records
    records = []
    for prompt_id, data in logprobs_by_prompt.items():
        # Skip if missing base policy data
        if "base_policy_logprob" not in data or data["base_policy_logprob"] is None:
            continue

        # Skip if no valid target policies
        target_logps = data.get("target_policy_logprobs", {})
        if not any(lp is not None for lp in target_logps.values()):
            continue

        # Get evaluation scores from base responses
        base_response_data = responses_by_prompt.get(prompt_id, {})
        metadata = {
            "prompt_id": data.get("prompt_id", prompt_id),
        }

        # Extract judge and oracle scores if available (top-level fields)
        judge_score = None
        oracle_label = None
        if "metadata" in base_response_data:
            response_metadata = base_response_data["metadata"]
            judge_score = response_metadata.get("judge_score")
            oracle_label = response_metadata.get("oracle_label")

        # Create record following core data model structure
        record = {
            "prompt": data["prompt"],
            "response": data["response"],
            "base_policy_logprob": data["base_policy_logprob"],
            "target_policy_logprobs": target_logps,
            # Note: No reward field - calibration happens during analysis
            "metadata": metadata,
        }

        # Add judge_score and oracle_label as top-level fields if present
        if judge_score is not None:
            record["judge_score"] = judge_score
        if oracle_label is not None:
            record["oracle_label"] = oracle_label

        records.append(record)

    # Track dropped records
    total_prompts = len(logprobs_by_prompt)
    dropped_base = sum(
        1
        for d in logprobs_by_prompt.values()
        if "base_policy_logprob" not in d or d.get("base_policy_logprob") is None
    )
    dropped_all_targets = sum(
        1
        for d in logprobs_by_prompt.values()
        if "base_policy_logprob" in d
        and d.get("base_policy_logprob") is not None
        and not any(
            lp is not None for lp in d.get("target_policy_logprobs", {}).values()
        )
    )

    print(f"Created {len(records)} complete records from {total_prompts} prompts")
    if dropped_base > 0:
        print(f"⚠️  Dropped {dropped_base} records with null base policy logprob")
    if dropped_all_targets > 0:
        print(f"⚠️  Dropped {dropped_all_targets} records with all target logprobs null")

    # Save dataset if output file is provided
    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        print(f"✓ Saved CJE dataset to {output_path}")

    # Warn if too few samples
    if len(records) < 10:
        print(
            f"\n⚠️  WARNING: Only {len(records)} samples in dataset. Minimum 10 recommended for reliable CJE analysis."
        )

    # Print summary statistics
    print("\nDataset summary:")
    print(f"  Total samples: {len(records)}")
    print(f"  Base policy: {base_policy}")
    print(f"  Target policies: {sorted(p for p in policies if p != base_policy)}")

    # Check how many records have evaluation scores
    with_judge = sum(1 for r in records if "judge_score" in r.get("metadata", {}))
    with_oracle = sum(1 for r in records if "oracle_label" in r.get("metadata", {}))
    print(f"\nEvaluation scores:")
    print(
        f"  Records with judge scores: {with_judge}/{len(records)} ({100*with_judge/len(records):.1f}%)"
    )
    print(
        f"  Records with oracle labels: {with_oracle}/{len(records)} ({100*with_oracle/len(records):.1f}%)"
    )

    valid_counts: Dict[str, int] = defaultdict(int)
    for record in records:
        for policy, logprob in record["target_policy_logprobs"].items():
            if logprob is not None:
                valid_counts[policy] += 1

    print("\nValid log probs per policy:")
    for policy, count in sorted(valid_counts.items()):
        print(f"  {policy}: {count}/{len(records)} ({100*count/len(records):.1f}%)")

    return records


def main() -> None:
    """Prepare complete CJE dataset."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logprobs-dir", default="data/logprobs", help="Directory with log prob files"
    )
    parser.add_argument(
        "--responses-dir",
        default="data/responses",
        help="Directory with response files",
    )
    parser.add_argument(
        "--output", default="data/cje_dataset.jsonl", help="Output CJE dataset"
    )
    parser.add_argument(
        "--base-policy", default="base", help="Name of base/behavior policy"
    )
    parser.add_argument(
        "--aggregation",
        choices=["mean", "median"],
        default="mean",
        help="Aggregation method for multiple passes (default: mean)",
    )

    args = parser.parse_args()

    # Prepare dataset (just combining data, no calibration)
    records = prepare_cje_dataset(
        logprobs_dir=args.logprobs_dir,
        responses_dir=args.responses_dir,
        output_file=args.output,
        base_policy=args.base_policy,
        aggregation=args.aggregation,
    )

    print(f"\n✓ Dataset ready for analysis with {len(records)} samples")


if __name__ == "__main__":
    main()
