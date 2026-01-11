#!/usr/bin/env python3
"""
Add binary LLM judge scores to MBPP response files.

Uses an LLM judge to make binary YES/NO predictions about whether code
passes all unit tests. This provides the binary judge output for comparing
continuous calibration (CJE) against binary misclassification adjustment
(Lee et al. / Rogan-Gladen).

Usage:
    # Score all policies
    python add_binary_judge.py

    # Score specific policy
    python add_binary_judge.py --policy base

    # Force rescore (overwrite existing scores)
    python add_binary_judge.py --force

    # Dry run to check parsing
    python add_binary_judge.py --dry-run --limit 10
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    POLICIES,
    RESPONSES_DIR,
    JUDGE_MODEL,
    JUDGE_TEMPERATURE,
    BATCH_SIZES,
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


# Binary judge prompt - simpler than continuous
BINARY_JUDGE_SYSTEM_PROMPT = """You are evaluating whether Python code correctly solves a programming problem.

Your task is to predict whether the code will pass ALL unit tests.
Consider:
- Does the function implement the required behavior?
- Does it handle edge cases correctly?
- Are there any syntax errors or obvious bugs?

Answer with a single word: YES or NO."""

BINARY_JUDGE_USER_TEMPLATE = """<problem>
{prompt}
</problem>

<code>
{response}
</code>

Does this code correctly solve the problem and pass all test cases?
Answer only YES or NO:"""


def parse_binary_response(text: str) -> Tuple[Optional[int], str]:
    """Parse a binary YES/NO response from the judge.

    Args:
        text: Raw response text from the judge

    Returns:
        Tuple of (parsed_value, parse_status)
        - parsed_value: 1 for YES, 0 for NO, None for parse failure
        - parse_status: 'yes', 'no', or 'parse_error'
    """
    text = text.strip().upper()

    # Handle common variations
    if text.startswith("YES") or text == "Y" or text == "1" or text == "TRUE":
        return 1, "yes"
    elif text.startswith("NO") or text == "N" or text == "0" or text == "FALSE":
        return 0, "no"
    else:
        # Try to find YES or NO in the response
        if re.search(r'\bYES\b', text):
            return 1, "yes"
        elif re.search(r'\bNO\b', text):
            return 0, "no"
        else:
            return None, "parse_error"


class BinaryJudge:
    """Binary YES/NO judge for MBPP code evaluation."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_prompt: str = BINARY_JUDGE_SYSTEM_PROMPT,
        user_template: str = BINARY_JUDGE_USER_TEMPLATE,
        temperature: float = 0.0,
    ):
        self.model = model.replace("openai/", "")
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.temperature = temperature
        self.client = openai.OpenAI()

        # Track parsing statistics
        self.parse_stats = {"yes": 0, "no": 0, "parse_error": 0, "api_error": 0}

    def score_single(self, prompt: str, response: str) -> Tuple[Optional[int], str]:
        """Score a single prompt/response pair with binary YES/NO.

        Returns:
            Tuple of (binary_score, raw_response)
            - binary_score: 1 for YES, 0 for NO, None if failed
            - raw_response: The raw text from the judge (for debugging)
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

            raw_text = result.choices[0].message.content.strip()
            parsed_value, parse_status = parse_binary_response(raw_text)
            self.parse_stats[parse_status] += 1

            return parsed_value, raw_text

        except Exception as e:
            self.parse_stats["api_error"] += 1
            return None, f"API_ERROR: {e}"

    def score_batch(
        self,
        prompts: List[str],
        responses: List[str],
        max_workers: int = 10,
    ) -> List[Tuple[Optional[int], str]]:
        """Score multiple prompt/response pairs in parallel.

        Returns:
            List of (binary_score, raw_response) tuples
        """
        results = [(None, "")] * len(prompts)

        def score_item(idx: int) -> Tuple[int, Optional[int], str]:
            binary_score, raw_text = self.score_single(prompts[idx], responses[idx])
            return idx, binary_score, raw_text

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(score_item, i) for i in range(len(prompts))]
            for future in as_completed(futures):
                idx, score, raw_text = future.result()
                results[idx] = (score, raw_text)

        return results


def load_responses_with_status(
    file_path: Path,
    score_field: str = "binary_judge",
) -> Tuple[List[Dict], List[int], int]:
    """Load responses and identify which need binary scoring.

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

            if "metadata" not in record:
                record["metadata"] = {}

            # Check if already scored
            if (
                score_field in record.get("metadata", {})
                and record["metadata"][score_field] is not None
            ):
                already_scored += 1
            else:
                if record.get("prompt") and record.get("response"):
                    indices_to_score.append(i)

    return records, indices_to_score, already_scored


def save_progress(records: List[Dict], output_file: Path) -> None:
    """Save progress atomically using temp file + rename."""
    temp_file = f"{output_file}.tmp.{os.getpid()}"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(temp_file, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    Path(temp_file).replace(output_file)


def add_binary_judge_to_file(
    input_file: Path,
    output_file: Optional[Path] = None,
    force_rescore: bool = False,
    batch_size: int = 50,
    save_every: int = 50,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Add binary judge scores to a response file.

    Args:
        input_file: Input JSONL file
        output_file: Output JSONL file (default: overwrite input)
        force_rescore: If True, rescore even if scores exist
        batch_size: Number of items to score in one batch
        save_every: Save progress every N scores
        dry_run: If True, don't make API calls, just check existing data
        limit: If set, only process this many samples

    Returns:
        Statistics dictionary
    """
    output_file = output_file or input_file

    print(f"\n{'='*60}")
    print(f"Adding binary judge scores to: {input_file.name}")
    print(f"{'='*60}")
    print(f"Judge model: {JUDGE_MODEL}")

    # Load existing data
    if force_rescore:
        print("Force rescore enabled - will overwrite existing binary scores")
        records = []
        with open(input_file, "r") as f:
            for line in f:
                record = json.loads(line)
                if "metadata" not in record:
                    record["metadata"] = {}
                if "binary_judge" in record.get("metadata", {}):
                    del record["metadata"]["binary_judge"]
                if "binary_judge_raw" in record.get("metadata", {}):
                    del record["metadata"]["binary_judge_raw"]
                records.append(record)
        indices_to_score = [
            i for i, r in enumerate(records)
            if r.get("prompt") and r.get("response")
        ]
        already_scored = 0
    else:
        check_file = output_file if output_file.exists() else input_file
        records, indices_to_score, already_scored = load_responses_with_status(
            check_file, "binary_judge"
        )

    total_records = len(records)
    valid_records = sum(1 for r in records if r.get("prompt") and r.get("response"))

    print(f"\nStatus:")
    print(f"  Total records:     {total_records}")
    print(f"  Valid records:     {valid_records}")
    print(f"  Already scored:    {already_scored}")
    print(f"  Need scoring:      {len(indices_to_score)}")

    if limit:
        indices_to_score = indices_to_score[:limit]
        print(f"  Limited to:        {limit}")

    if not indices_to_score:
        print("\nAll records already have binary scores!")
        return {
            "total": total_records,
            "scored": already_scored,
            "failed": 0,
            "parse_errors": 0,
        }

    if dry_run:
        print("\n[DRY RUN] Would score these samples:")
        for idx in indices_to_score[:5]:
            print(f"  - Record {idx}: prompt_id={records[idx].get('prompt_id', 'N/A')}")
        print(f"  ... and {len(indices_to_score) - 5} more")
        return {"dry_run": True, "would_score": len(indices_to_score)}

    # Create binary judge
    judge = BinaryJudge(
        model=JUDGE_MODEL,
        temperature=JUDGE_TEMPERATURE,
    )

    # Process in batches
    print(f"\nBinary scoring {len(indices_to_score)} records...")
    failed_count = 0
    scores_added = 0
    parse_errors = 0

    with tqdm(total=len(indices_to_score), desc="Binary judge", unit="score") as pbar:
        for batch_start in range(0, len(indices_to_score), batch_size):
            batch_end = min(batch_start + batch_size, len(indices_to_score))
            batch_indices = indices_to_score[batch_start:batch_end]

            prompts = [records[idx]["prompt"] for idx in batch_indices]
            responses = [records[idx]["response"] for idx in batch_indices]

            try:
                results = judge.score_batch(prompts, responses)

                for i, record_idx in enumerate(batch_indices):
                    binary_score, raw_text = results[i]
                    records[record_idx]["metadata"]["binary_judge"] = binary_score
                    records[record_idx]["metadata"]["binary_judge_raw"] = raw_text
                    records[record_idx]["metadata"]["binary_judge_model"] = JUDGE_MODEL

                    if binary_score is not None:
                        scores_added += 1
                    else:
                        failed_count += 1
                        if "parse_error" in raw_text.lower() or raw_text.startswith("API_ERROR"):
                            parse_errors += 1

                    pbar.update(1)

            except Exception as e:
                print(f"\nBatch failed: {e}")
                failed_count += len(batch_indices)
                pbar.update(len(batch_indices))

            # Save progress periodically
            if scores_added > 0 and scores_added % save_every == 0:
                save_progress(records, output_file)
                pbar.set_postfix({
                    "saved": scores_added,
                    "failed": failed_count,
                    "yes%": f"{judge.parse_stats['yes']/(judge.parse_stats['yes']+judge.parse_stats['no']+0.001)*100:.1f}"
                })

    # Final save
    save_progress(records, output_file)

    # Statistics
    print(f"\n{'='*60}")
    print("Binary Scoring Complete!")
    print(f"{'='*60}")
    print(f"  New scores added:  {scores_added}")
    print(f"  Failed to score:   {failed_count}")
    print(f"  Parse errors:      {parse_errors}")
    print(f"  Total scored:      {already_scored + scores_added}")
    print(f"  Output saved to:   {output_file}")

    # Parse statistics from judge
    print(f"\nJudge Response Statistics:")
    print(f"  YES responses:     {judge.parse_stats['yes']}")
    print(f"  NO responses:      {judge.parse_stats['no']}")
    print(f"  Parse errors:      {judge.parse_stats['parse_error']}")
    print(f"  API errors:        {judge.parse_stats['api_error']}")

    # Binary score distribution
    all_binary = [
        r["metadata"].get("binary_judge")
        for r in records
        if r.get("metadata", {}).get("binary_judge") is not None
    ]

    if all_binary:
        yes_rate = sum(all_binary) / len(all_binary)
        print(f"\nBinary Score Distribution:")
        print(f"  YES rate:          {yes_rate:.1%} ({sum(all_binary)}/{len(all_binary)})")
        print(f"  NO rate:           {1-yes_rate:.1%} ({len(all_binary)-sum(all_binary)}/{len(all_binary)})")

        # Compare to continuous judge if available
        continuous_scores = [
            r["metadata"].get("judge_score")
            for r in records
            if r.get("metadata", {}).get("judge_score") is not None
        ]
        if continuous_scores:
            import numpy as np
            print(f"\nComparison to Continuous Judge:")
            print(f"  Continuous mean:   {np.mean(continuous_scores):.3f}")
            print(f"  Binary YES rate:   {yes_rate:.3f}")

    return {
        "total": total_records,
        "scored": already_scored + scores_added,
        "failed": failed_count,
        "parse_errors": parse_errors,
        "parse_stats": judge.parse_stats,
    }


def add_binary_judge_all_policies(
    policies: Optional[List[str]] = None,
    force_rescore: bool = False,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Add binary judge scores to all (or specified) policies."""
    results = {}
    policy_list = policies if policies else list(POLICIES.keys())

    print(f"\n{'#'*60}")
    print(f"# Adding Binary Judge Scores")
    print(f"# Policies: {', '.join(policy_list)}")
    print(f"# Judge: {JUDGE_MODEL}")
    print(f"{'#'*60}")

    for policy_name in policy_list:
        response_file = RESPONSES_DIR / f"{policy_name}_responses.jsonl"

        if not response_file.exists():
            print(f"\nWarning: Response file not found for '{policy_name}'")
            results[policy_name] = {"error": "File not found"}
            continue

        stats = add_binary_judge_to_file(
            input_file=response_file,
            force_rescore=force_rescore,
            dry_run=dry_run,
            limit=limit,
        )
        results[policy_name] = stats

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")

    total_yes = 0
    total_no = 0

    for policy, stats in results.items():
        if "error" in stats:
            print(f"  ✗ {policy}: {stats['error']}")
        elif stats.get("dry_run"):
            print(f"  ○ {policy}: Would score {stats['would_score']}")
        else:
            parse_stats = stats.get("parse_stats", {})
            yes_count = parse_stats.get("yes", 0)
            no_count = parse_stats.get("no", 0)
            total_yes += yes_count
            total_no += no_count
            yes_rate = yes_count / (yes_count + no_count + 0.001)
            print(f"  ✓ {policy}: {stats['scored']}/{stats['total']} scored, {yes_rate:.1%} YES")

    if total_yes + total_no > 0:
        print(f"\nOverall YES rate: {total_yes/(total_yes+total_no):.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Add binary LLM judge scores to MBPP response files"
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
        "--dry-run",
        action="store_true",
        help="Don't make API calls, just check what would be scored",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of samples to score (for testing)",
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

    # Determine which policies to process
    if args.policy:
        policies = [args.policy]
    elif args.policies:
        policies = args.policies
    else:
        policies = None

    results = add_binary_judge_all_policies(
        policies=policies,
        force_rescore=args.force,
        dry_run=args.dry_run,
        limit=args.limit,
    )

    # Return non-zero if any failed
    for stats in results.values():
        if "error" in stats or stats.get("failed", 0) > 0:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
