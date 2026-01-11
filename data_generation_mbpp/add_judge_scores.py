#!/usr/bin/env python3
"""
Add LLM judge scores to MBPP response files.

Uses an LLM judge to predict the probability that each code solution
will pass all unit tests, providing the surrogate score for CJE.

Usage:
    # Score all policies
    python add_judge_scores.py

    # Score specific policy
    python add_judge_scores.py --policy base

    # Force rescore (overwrite existing scores)
    python add_judge_scores.py --force
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    POLICIES,
    RESPONSES_DIR,
    JUDGE_MODEL,
    JUDGE_TEMPERATURE,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    BATCH_SIZES,
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


class OpenAIJudge:
    """Simple OpenAI-based judge for MBPP code evaluation."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_prompt: str = "",
        user_template: str = "{prompt}\n{response}",
        temperature: float = 0.0,
    ):
        self.model = model.replace("openai/", "")  # Strip prefix if present
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.temperature = temperature
        self.client = openai.OpenAI()

    def score_single(self, prompt: str, response: str) -> Optional[float]:
        """Score a single prompt/response pair.

        Returns:
            Score from 0.0 to 1.0, or None if failed
        """
        user_content = self.user_template.format(prompt=prompt, response=response)

        try:
            result = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=self.temperature,
                max_tokens=10,
            )

            # Extract score from response
            text = result.choices[0].message.content.strip()

            # Parse integer score (0-100)
            match = re.search(r'\b(\d+)\b', text)
            if match:
                score = int(match.group(1))
                # Normalize to 0-1 range
                return min(max(score / 100.0, 0.0), 1.0)

            return None

        except Exception as e:
            print(f"Error scoring: {e}")
            return None

    def score_batch(
        self,
        prompts: List[str],
        responses: List[str],
        max_workers: int = 10,
    ) -> List[Optional[float]]:
        """Score multiple prompt/response pairs in parallel.

        Returns:
            List of scores (0.0-1.0) or None for failures
        """
        scores = [None] * len(prompts)

        def score_item(idx: int) -> tuple[int, Optional[float]]:
            return idx, self.score_single(prompts[idx], responses[idx])

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(score_item, i) for i in range(len(prompts))]
            for future in as_completed(futures):
                idx, score = future.result()
                scores[idx] = score

        return scores


def load_responses_with_status(
    file_path: Path,
    score_field: str = "judge_score",
) -> tuple[List[Dict], List[int], int]:
    """Load responses and identify which need scoring.

    Args:
        file_path: Path to JSONL file
        score_field: Field name in metadata for scores

    Returns:
        Tuple of (records, indices_to_score, already_scored_count)
    """
    records = []
    indices_to_score = []
    already_scored = 0

    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            records.append(record)

            # Ensure metadata exists
            if "metadata" not in record:
                record["metadata"] = {}

            # Check if already scored
            if (
                score_field in record.get("metadata", {})
                and record["metadata"][score_field] is not None
            ):
                already_scored += 1
            else:
                # Check if we have prompt and response
                if record.get("prompt") and record.get("response"):
                    indices_to_score.append(i)

    return records, indices_to_score, already_scored


def save_progress(
    records: List[Dict],
    output_file: Path,
) -> None:
    """Save progress atomically using temp file + rename."""
    import os

    temp_file = f"{output_file}.tmp.{os.getpid()}"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(temp_file, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    Path(temp_file).replace(output_file)


def add_judge_scores_to_file(
    input_file: Path,
    output_file: Optional[Path] = None,
    force_rescore: bool = False,
    batch_size: int = BATCH_SIZES["judge_scoring"],
    save_every: int = 50,
) -> Dict[str, Any]:
    """Add judge scores to a response file.

    Args:
        input_file: Input JSONL file
        output_file: Output JSONL file (default: overwrite input)
        force_rescore: If True, rescore even if scores exist
        batch_size: Number of items to score in one API call
        save_every: Save progress every N scores

    Returns:
        Statistics dictionary
    """
    output_file = output_file or input_file

    print(f"\n{'='*60}")
    print(f"Adding judge scores to: {input_file.name}")
    print(f"{'='*60}")
    print(f"Judge model: {JUDGE_MODEL}")

    # Load existing data
    if force_rescore:
        print("Force rescore enabled - will overwrite existing scores")
        records = []
        with open(input_file, "r") as f:
            for line in f:
                record = json.loads(line)
                if "metadata" not in record:
                    record["metadata"] = {}
                if "judge_score" in record.get("metadata", {}):
                    del record["metadata"]["judge_score"]
                records.append(record)
        indices_to_score = [
            i for i, r in enumerate(records)
            if r.get("prompt") and r.get("response")
        ]
        already_scored = 0
    else:
        check_file = output_file if output_file.exists() else input_file
        records, indices_to_score, already_scored = load_responses_with_status(
            check_file, "judge_score"
        )

    total_records = len(records)
    valid_records = sum(1 for r in records if r.get("prompt") and r.get("response"))

    print(f"\nStatus:")
    print(f"  Total records:     {total_records}")
    print(f"  Valid records:     {valid_records}")
    print(f"  Already scored:    {already_scored}")
    print(f"  Need scoring:      {len(indices_to_score)}")

    if not indices_to_score:
        print("\nAll records already scored!")
        if output_file != input_file and not output_file.exists():
            save_progress(records, output_file)
            print(f"Saved to {output_file}")
        return {
            "total": total_records,
            "scored": already_scored,
            "failed": 0,
        }

    # Create evaluator with MBPP-specific prompt
    judge = OpenAIJudge(
        model=JUDGE_MODEL,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_template=JUDGE_USER_TEMPLATE,
        temperature=JUDGE_TEMPERATURE,
    )

    # Process in batches
    print(f"\nScoring {len(indices_to_score)} records...")
    failed_count = 0
    scores_added = 0

    with tqdm(total=len(indices_to_score), desc="Judge scoring", unit="score") as pbar:
        for batch_start in range(0, len(indices_to_score), batch_size):
            batch_end = min(batch_start + batch_size, len(indices_to_score))
            batch_indices = indices_to_score[batch_start:batch_end]

            # Collect prompts and responses
            prompts = []
            responses = []
            for idx in batch_indices:
                prompts.append(records[idx]["prompt"])
                responses.append(records[idx]["response"])

            # Score the batch
            try:
                scores = judge.score_batch(prompts, responses)

                # Update records with scores
                for i, record_idx in enumerate(batch_indices):
                    score = scores[i]
                    records[record_idx]["metadata"]["judge_score"] = score
                    records[record_idx]["metadata"]["judge_model"] = JUDGE_MODEL

                    if score is not None:
                        scores_added += 1
                    else:
                        failed_count += 1

                    pbar.update(1)

            except Exception as e:
                print(f"\nBatch failed: {e}")
                failed_count += len(batch_indices)
                pbar.update(len(batch_indices))

            # Save progress periodically
            if scores_added > 0 and scores_added % save_every == 0:
                save_progress(records, output_file)
                pbar.set_postfix({"saved": scores_added, "failed": failed_count})

    # Final save
    save_progress(records, output_file)

    # Statistics
    print(f"\n{'='*60}")
    print("Scoring Complete!")
    print(f"{'='*60}")
    print(f"  New scores added:  {scores_added}")
    print(f"  Failed to score:   {failed_count}")
    print(f"  Total scored:      {already_scored + scores_added}")
    print(f"  Output saved to:   {output_file}")

    # Score statistics
    all_scores = [
        r["metadata"].get("judge_score")
        for r in records
        if r.get("metadata", {}).get("judge_score") is not None
    ]

    if all_scores:
        import numpy as np
        print(f"\nScore Statistics:")
        print(f"  Mean:   {np.mean(all_scores):.3f}")
        print(f"  Median: {np.median(all_scores):.3f}")
        print(f"  Std:    {np.std(all_scores):.3f}")
        print(f"  Range:  [{np.min(all_scores):.3f}, {np.max(all_scores):.3f}]")

    return {
        "total": total_records,
        "scored": already_scored + scores_added,
        "failed": failed_count,
    }


def add_judge_scores_all_policies(
    policies: Optional[List[str]] = None,
    force_rescore: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Add judge scores to all (or specified) policies.

    Args:
        policies: Optional list of specific policies
        force_rescore: If True, rescore even if scores exist

    Returns:
        Dictionary mapping policy names to statistics
    """
    results = {}
    policy_list = policies if policies else list(POLICIES.keys())

    print(f"\n{'#'*60}")
    print(f"# Adding Judge Scores")
    print(f"# Policies: {', '.join(policy_list)}")
    print(f"# Judge: {JUDGE_MODEL}")
    print(f"{'#'*60}")

    for policy_name in policy_list:
        response_file = RESPONSES_DIR / f"{policy_name}_responses.jsonl"

        if not response_file.exists():
            print(f"\nWarning: Response file not found for '{policy_name}'")
            print(f"  Expected: {response_file}")
            results[policy_name] = {"error": "File not found"}
            continue

        stats = add_judge_scores_to_file(
            input_file=response_file,
            force_rescore=force_rescore,
        )
        results[policy_name] = stats

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for policy, stats in results.items():
        if "error" in stats:
            print(f"  ✗ {policy}: {stats['error']}")
        else:
            print(f"  ✓ {policy}: {stats['scored']}/{stats['total']} scored")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Add LLM judge scores to MBPP response files"
    )
    parser.add_argument(
        "--policy", "-p",
        type=str,
        help="Specific policy to score (default: all policies)",
    )
    parser.add_argument(
        "--policies",
        type=str,
        nargs="+",
        help="List of policies to score",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force rescore even if scores exist",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Score a specific file instead of policies",
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

    if args.file:
        # Score specific file
        if not args.file.exists():
            print(f"Error: File not found: {args.file}")
            return 1
        stats = add_judge_scores_to_file(
            input_file=args.file,
            force_rescore=args.force,
        )
        return 0 if stats["failed"] == 0 else 1

    # Determine which policies to process
    if args.policy:
        policies = [args.policy]
    elif args.policies:
        policies = args.policies
    else:
        policies = None  # All policies

    results = add_judge_scores_all_policies(
        policies=policies,
        force_rescore=args.force,
    )

    # Return non-zero if any failed
    for stats in results.values():
        if "error" in stats or stats.get("failed", 0) > 0:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
