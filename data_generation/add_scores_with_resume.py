#!/usr/bin/env python3
"""
Enhanced scoring script with resume capability and progress tracking.

This module provides functions to add judge scores and oracle labels
with proper resume logic for interrupted runs.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys
from tqdm import tqdm

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from evaluation_utils import (
    FireworksEvaluator,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_ORACLE_MODEL,
)
from experiment_config import BATCH_SIZES


def load_existing_scores(
    file_path: str, score_field: str = "judge_score"
) -> Tuple[List[Dict], List[int], int]:
    """Load existing records and identify which ones need scoring.

    Args:
        file_path: Path to JSONL file
        score_field: Field name to check in metadata

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
    records: List[Dict], output_file: str, temp_suffix: str = ".tmp"
) -> None:
    """Save progress atomically using temp file + rename.

    Args:
        records: Records to save
        output_file: Output file path
        temp_suffix: Suffix for temp file
    """
    import os

    # Include PID to avoid collision with parallel runs
    temp_file = f"{output_file}{temp_suffix}.{os.getpid()}"
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file
    with open(temp_file, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    # Atomic rename
    Path(temp_file).replace(output_path)


def add_scores_with_resume(
    input_file: str,
    output_file: str,
    evaluator: Any,
    score_field: str = "judge_score",
    desc: str = "Scoring",
    batch_size: int = BATCH_SIZES["judge_scoring"],
    save_every: int = BATCH_SIZES[
        "judge_scoring"
    ],  # Default to batch_size for consistent saves
    force_rescore: bool = False,
) -> Dict[str, Any]:
    """Add scores with resume capability and progress tracking.

    Args:
        input_file: Input JSONL file
        output_file: Output JSONL file (can be same as input for in-place update)
        evaluator: Evaluator instance (FireworksEvaluator)
        score_field: Field name in metadata for scores
        desc: Description for progress bar
        batch_size: Number of items to score in one API call
        save_every: Save progress every N scores
        force_rescore: If True, rescore even if scores exist

    Returns:
        Dictionary with statistics
    """
    print(f"\n{'='*60}")
    print(f"Adding {score_field} to dataset")
    print(f"{'='*60}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Model:  {evaluator.model}")

    # Load existing data and check what needs scoring
    if force_rescore:
        print("Force rescore enabled - will overwrite existing scores")
        records = []
        with open(input_file, "r") as f:
            for line in f:
                record = json.loads(line)
                if "metadata" not in record:
                    record["metadata"] = {}
                # Clear existing score
                if score_field in record.get("metadata", {}):
                    del record["metadata"][score_field]
                records.append(record)

        indices_to_score = [
            i for i, r in enumerate(records) if r.get("prompt") and r.get("response")
        ]
        already_scored = 0
    else:
        # Use output file if it exists, otherwise input file
        check_file = output_file if Path(output_file).exists() else input_file
        records, indices_to_score, already_scored = load_existing_scores(
            check_file, score_field
        )

    total_records = len(records)
    valid_records = sum(1 for r in records if r.get("prompt") and r.get("response"))

    print(f"\n📊 Status:")
    print(f"  Total records:     {total_records}")
    print(f"  Valid records:     {valid_records}")
    print(f"  Already scored:    {already_scored}")
    print(f"  Need scoring:      {len(indices_to_score)}")

    if not indices_to_score:
        print("\n✅ All records already scored!")
        if output_file != input_file and not Path(output_file).exists():
            # Still need to save output file
            save_progress(records, output_file)
            print(f"✓ Saved to {output_file}")
        return {
            "total": total_records,
            "scored": already_scored,
            "skipped": 0,
            "failed": 0,
        }

    # Prepare for scoring
    print(f"\n🚀 Starting scoring of {len(indices_to_score)} records...")
    print(f"  Batch size: {batch_size}")
    print(f"  Save every: {save_every} scores")

    failed_count = 0
    scores_added = 0

    # Process in batches with progress bar
    with tqdm(total=len(indices_to_score), desc=desc, unit="score") as pbar:
        for batch_start in range(0, len(indices_to_score), batch_size):
            batch_end = min(batch_start + batch_size, len(indices_to_score))
            batch_indices = indices_to_score[batch_start:batch_end]

            # Collect prompts and responses for this batch
            prompts = []
            responses = []
            for idx in batch_indices:
                prompts.append(records[idx]["prompt"])
                responses.append(records[idx]["response"])

            # Score the batch
            try:
                result = evaluator.score_batch(
                    prompts,
                    responses,
                    show_progress=False,  # We're using our own progress bar
                    skip_failures=True,
                )

                # Update records with scores
                for i, record_idx in enumerate(batch_indices):
                    score = result.scores[i] if result.scores[i] is not None else None
                    records[record_idx]["metadata"][score_field] = score

                    # Also store metadata about the scoring (model name, etc)
                    if result.metadata and i < len(result.metadata):
                        score_metadata = result.metadata[i]
                        if score_metadata and isinstance(score_metadata, dict):
                            # Store judge_model if present
                            if "judge_model" in score_metadata:
                                records[record_idx]["metadata"]["judge_model"] = (
                                    score_metadata["judge_model"]
                                )

                    if score is not None:
                        scores_added += 1
                    else:
                        failed_count += 1

                    pbar.update(1)

            except Exception as e:
                print(f"\n❌ Batch failed: {e}")
                failed_count += len(batch_indices)
                pbar.update(len(batch_indices))

            # Save progress periodically
            # Save progress periodically based on save_every parameter
            if scores_added > 0 and scores_added % save_every == 0:
                save_progress(records, output_file)
                pbar.set_postfix({"saved": scores_added, "failed": failed_count})

    # Final save
    save_progress(records, output_file)

    # Print summary
    print(f"\n{'='*60}")
    print(f"✅ Scoring Complete!")
    print(f"{'='*60}")
    print(f"  New scores added:  {scores_added}")
    print(f"  Failed to score:   {failed_count}")
    print(f"  Total scored:      {already_scored + scores_added}")
    print(f"  Output saved to:   {output_file}")

    # Print score statistics
    all_scores = [
        r["metadata"].get(score_field)
        for r in records
        if r.get("metadata", {}).get(score_field) is not None
    ]

    if all_scores:
        import numpy as np

        print(f"\n📈 Score Statistics:")
        print(f"  Mean:   {np.mean(all_scores):.3f}")
        print(f"  Median: {np.median(all_scores):.3f}")
        print(f"  Std:    {np.std(all_scores):.3f}")
        print(f"  Range:  [{np.min(all_scores):.3f}, {np.max(all_scores):.3f}]")

    return {
        "total": total_records,
        "scored": already_scored + scores_added,
        "skipped": total_records - valid_records,
        "failed": failed_count,
    }


def add_judge_scores_with_resume(
    input_file: str, output_file: str, model: str = DEFAULT_JUDGE_MODEL, **kwargs: Any
) -> Dict[str, Any]:
    """Add judge scores with resume capability.

    Args:
        input_file: Input JSONL file
        output_file: Output JSONL file
        model: Judge model to use
        **kwargs: Additional arguments for add_scores_with_resume

    Returns:
        Statistics dictionary
    """
    evaluator = FireworksEvaluator(model=model)
    return add_scores_with_resume(
        input_file,
        output_file,
        evaluator,
        score_field="judge_score",
        desc="Judge scoring",
        **kwargs,
    )


def add_oracle_labels_with_resume(
    input_file: str, output_file: str, model: str = DEFAULT_ORACLE_MODEL, **kwargs: Any
) -> Dict[str, Any]:
    """Add oracle labels with resume capability.

    Args:
        input_file: Input JSONL file
        output_file: Output JSONL file
        model: Oracle model to use
        **kwargs: Additional arguments for add_scores_with_resume

    Returns:
        Statistics dictionary
    """
    evaluator = FireworksEvaluator(model=model)
    return add_scores_with_resume(
        input_file,
        output_file,
        evaluator,
        score_field="oracle_label",
        desc="Oracle evaluation",
        **kwargs,
    )


def main() -> int:
    """CLI for adding scores with resume capability."""
    import argparse

    parser = argparse.ArgumentParser(description="Add scores with resume capability")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("-o", "--output", help="Output file (default: overwrite input)")
    parser.add_argument(
        "--type", choices=["judge", "oracle"], default="judge", help="Type of scoring"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZES["judge_scoring"],
        help=f"Batch size for API calls (default: {BATCH_SIZES['judge_scoring']})",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=BATCH_SIZES["judge_scoring"],
        help=f"Save progress every N scores (default: {BATCH_SIZES['judge_scoring']}, matches batch size)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force rescore even if scores exist"
    )

    args = parser.parse_args()

    output_file = args.output or args.input

    if args.type == "judge":
        stats = add_judge_scores_with_resume(
            args.input,
            output_file,
            batch_size=args.batch_size,
            save_every=args.save_every,
            force_rescore=args.force,
        )
    else:
        stats = add_oracle_labels_with_resume(
            args.input,
            output_file,
            batch_size=args.batch_size,
            save_every=args.save_every,
            force_rescore=args.force,
        )

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
