#!/usr/bin/env python3
"""
Extract results from Inspect MBPP evaluation logs.

Parses Inspect JSON/eval logs and converts them to the JSONL format
required for CJE analysis.

Usage:
    # Extract all policies
    python extract_inspect_results.py

    # Extract specific policy
    python extract_inspect_results.py --policy base

    # Specify custom input/output directories
    python extract_inspect_results.py --input-dir data_mbpp/raw --output-dir data_mbpp/responses
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from glob import glob

from config import POLICIES, DATA_DIR, RAW_DIR, RESPONSES_DIR


def find_latest_log(policy_dir: Path) -> Optional[Path]:
    """Find the most recent Inspect log file in a directory.

    Args:
        policy_dir: Directory containing Inspect logs

    Returns:
        Path to the most recent log file, or None if not found
    """
    # Inspect can create .json or .eval files
    patterns = ["*.json", "*.eval"]

    log_files = []
    for pattern in patterns:
        log_files.extend(policy_dir.glob(pattern))

    if not log_files:
        return None

    # Return most recently modified
    return max(log_files, key=lambda p: p.stat().st_mtime)


def load_inspect_log(log_path: Path) -> Dict[str, Any]:
    """Load an Inspect log file.

    Handles both JSON and .eval formats.

    Args:
        log_path: Path to log file

    Returns:
        Parsed log data as dictionary
    """
    if log_path.suffix == ".eval":
        # .eval files are binary format - need to use Inspect's API
        try:
            from inspect_ai.log import read_eval_log
            log = read_eval_log(str(log_path))
            # Convert to dict-like structure
            return {
                "status": log.status,
                "eval": {
                    "task": log.eval.task,
                    "model": log.eval.model,
                },
                "samples": [
                    {
                        "id": s.id,
                        "input": s.input,
                        "output": s.output,
                        "target": s.target,
                        "score": s.score,
                        "metadata": s.metadata,
                    }
                    for s in log.samples
                ] if log.samples else [],
                "results": log.results,
            }
        except ImportError:
            print("Warning: inspect_ai not available, cannot read .eval files")
            print("Install with: pip install inspect-ai")
            raise
    else:
        # JSON format
        with open(log_path) as f:
            return json.load(f)


def extract_code_from_response(response: str) -> str:
    """Extract Python code from a model response.

    Handles markdown code blocks and plain code.

    Args:
        response: Raw model response

    Returns:
        Extracted Python code
    """
    # Check for markdown code blocks
    if "```python" in response:
        # Extract code between ```python and ```
        parts = response.split("```python")
        if len(parts) > 1:
            code_part = parts[1].split("```")[0]
            return code_part.strip()

    if "```" in response:
        # Generic code block
        parts = response.split("```")
        if len(parts) > 1:
            return parts[1].strip()

    # Return as-is (already plain code)
    return response.strip()


def extract_results_from_log(
    log_data: Dict[str, Any],
    policy_name: str,
) -> List[Dict[str, Any]]:
    """Extract results from parsed Inspect log.

    Args:
        log_data: Parsed log data
        policy_name: Name of the policy

    Returns:
        List of result records in CJE format
    """
    results = []
    samples = log_data.get("samples", [])

    for sample in samples:
        # Extract sample ID (MBPP problem ID)
        sample_id = sample.get("id", "")
        prompt_id = f"mbpp_{sample_id}"

        # Extract prompt (the problem description)
        input_data = sample.get("input", {})
        if isinstance(input_data, dict):
            # Input might be nested
            prompt = input_data.get("prompt", "") or input_data.get("text", "")
            if not prompt and "messages" in input_data:
                # Extract from messages format
                messages = input_data["messages"]
                for msg in messages:
                    if msg.get("role") == "user":
                        prompt = msg.get("content", "")
                        break
        elif isinstance(input_data, str):
            prompt = input_data
        else:
            prompt = str(input_data)

        # Extract response (generated code)
        output_data = sample.get("output", {})
        if isinstance(output_data, dict):
            response = output_data.get("completion", "") or output_data.get("text", "")
            if not response and "choices" in output_data:
                # OpenAI-style format
                choices = output_data["choices"]
                if choices:
                    response = choices[0].get("message", {}).get("content", "")
        elif isinstance(output_data, str):
            response = output_data
        else:
            response = str(output_data)

        # Clean up response - extract code if wrapped in markdown
        code = extract_code_from_response(response)

        # Extract score (oracle label from test execution)
        # Inspect MBPP scorer returns "C" for correct, "I" for incorrect
        # Score is nested in sample['scores']['verify']['value']
        scores_data = sample.get("scores", {})
        if isinstance(scores_data, dict):
            # Get the verify scorer result
            verify_score = scores_data.get("verify", {})
            if isinstance(verify_score, dict):
                score_value = verify_score.get("value", None)
            else:
                score_value = verify_score
        else:
            # Fallback to old format
            score_data = sample.get("score", {})
            if isinstance(score_data, dict):
                score_value = score_data.get("value", None)
            else:
                score_value = score_data

        # Convert to binary oracle label
        if score_value == "C" or score_value == 1 or score_value == 1.0:
            oracle_label = 1.0
            test_passed = True
        elif score_value == "I" or score_value == 0 or score_value == 0.0:
            oracle_label = 0.0
            test_passed = False
        else:
            # Unknown score - mark as failed
            oracle_label = 0.0
            test_passed = False

        # Build result record
        result = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "response": code,
            "metadata": {
                "oracle_label": oracle_label,
                "test_passed": test_passed,
                "policy": policy_name,
                "mbpp_id": sample_id,
                "response_length": len(code.split()),  # Word count for covariate
                "raw_response": response,  # Keep original for debugging
            },
        }

        # Add any test case info if available
        if sample.get("target"):
            result["metadata"]["test_cases"] = sample["target"]

        results.append(result)

    return results


def extract_policy_results(
    policy_name: str,
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Extract results for a single policy.

    Args:
        policy_name: Name of the policy
        input_dir: Directory containing raw Inspect logs
        output_dir: Directory for output JSONL files

    Returns:
        Path to output file, or None if failed
    """
    input_dir = input_dir or RAW_DIR
    output_dir = output_dir or RESPONSES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_log_dir = input_dir / policy_name

    if not policy_log_dir.exists():
        print(f"Warning: No log directory found for policy '{policy_name}'")
        print(f"  Expected: {policy_log_dir}")
        return None

    # Find the log file
    log_path = find_latest_log(policy_log_dir)
    if not log_path:
        print(f"Warning: No log files found in {policy_log_dir}")
        return None

    print(f"\nExtracting results for policy: {policy_name}")
    print(f"  Input: {log_path}")

    # Load and parse log
    try:
        log_data = load_inspect_log(log_path)
    except Exception as e:
        print(f"  Error loading log: {e}")
        return None

    # Extract results
    results = extract_results_from_log(log_data, policy_name)

    if not results:
        print(f"  Warning: No results extracted from log")
        return None

    # Save to JSONL
    output_path = output_dir / f"{policy_name}_responses.jsonl"
    with open(output_path, "w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    # Statistics
    n_passed = sum(1 for r in results if r["metadata"]["test_passed"])
    n_total = len(results)
    pass_rate = n_passed / n_total if n_total > 0 else 0

    print(f"  Output: {output_path}")
    print(f"  Records: {n_total}")
    print(f"  Pass rate: {n_passed}/{n_total} ({pass_rate:.1%})")

    return output_path


def extract_all_policies(
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    policies: Optional[List[str]] = None,
) -> Dict[str, Optional[Path]]:
    """Extract results for all (or specified) policies.

    Args:
        input_dir: Directory containing raw Inspect logs
        output_dir: Directory for output JSONL files
        policies: Optional list of specific policies to process

    Returns:
        Dictionary mapping policy names to output paths
    """
    results = {}
    policy_list = policies if policies else list(POLICIES.keys())

    print(f"\n{'#'*60}")
    print(f"# Extracting MBPP Results")
    print(f"# Policies: {', '.join(policy_list)}")
    print(f"{'#'*60}")

    for policy_name in policy_list:
        output_path = extract_policy_results(
            policy_name,
            input_dir=input_dir,
            output_dir=output_dir,
        )
        results[policy_name] = output_path

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for policy, path in results.items():
        icon = "✓" if path else "✗"
        status = str(path) if path else "failed"
        print(f"  {icon} {policy}: {status}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract results from Inspect MBPP evaluation logs"
    )
    parser.add_argument(
        "--policy", "-p",
        type=str,
        help="Specific policy to extract (default: all policies)",
    )
    parser.add_argument(
        "--policies",
        type=str,
        nargs="+",
        help="List of policies to extract",
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        default=RAW_DIR,
        help=f"Input directory with Inspect logs (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=RESPONSES_DIR,
        help=f"Output directory for JSONL files (default: {RESPONSES_DIR})",
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
        policies = None  # All policies

    results = extract_all_policies(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        policies=policies,
    )

    # Return non-zero if any failed
    if None in results.values():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
