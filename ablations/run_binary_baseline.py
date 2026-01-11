#!/usr/bin/env python3
"""
Binary baseline experiments for MBPP ablation study.

Compares continuous CJE calibration against binary misclassification adjustment:
- naive-direct (continuous, no calibration)
- direct (continuous, isotonic calibration)
- binary-naive (binary, no calibration)
- binary-adjusted (binary, Rogan-Gladen)

Runs the same experimental grid as MBPP ablation:
- Sample sizes: 200, 400, 600, 800, 974
- Oracle coverages: 5%, 10%, 25%, 50%, 100%
- Seeds: 50

Usage:
    # Run all experiments
    python run_binary_baseline.py

    # Run with fewer workers
    python run_binary_baseline.py --workers 4

    # Test with single seed
    python run_binary_baseline.py --test
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from binary_baseline import (
    estimate_confusion_stats,
    binary_naive_estimate,
    binary_adjusted_estimate,
    confusion_matrix_transport_audit,
    ConfusionStats,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data_mbpp"
RESULTS_DIR = BASE_DIR / "results_mbpp"
RESULTS_FILE = RESULTS_DIR / "binary_baseline.jsonl"

# Experimental grid (same as main MBPP ablation)
SAMPLE_SIZES = [200, 400, 600, 800, 974]
ORACLE_COVERAGES = [0.05, 0.10, 0.25, 0.50, 1.00]
SEEDS = list(range(50))
TARGET_POLICIES = ["enhanced", "mini", "terse", "unhelpful"]


def load_policy_data(policy: str) -> pd.DataFrame:
    """Load response data for a policy.

    Returns DataFrame with columns:
    - prompt_id
    - oracle_label
    - judge_score (continuous)
    - binary_judge (binary)
    """
    response_file = DATA_DIR / "responses" / f"{policy}_responses.jsonl"

    records = []
    with open(response_file, "r") as f:
        for line in f:
            data = json.loads(line)
            records.append({
                "prompt_id": data["prompt_id"],
                "oracle_label": data["metadata"]["oracle_label"],
                "judge_score": data["metadata"].get("judge_score"),
                "binary_judge": data["metadata"].get("binary_judge"),
            })

    return pd.DataFrame(records)


def load_all_policies() -> Dict[str, pd.DataFrame]:
    """Load data for all policies."""
    policies = ["base"] + TARGET_POLICIES
    data = {}

    for policy in policies:
        try:
            df = load_policy_data(policy)
            data[policy] = df
            logger.info(f"Loaded {len(df)} samples for {policy}")
        except Exception as e:
            logger.error(f"Failed to load {policy}: {e}")

    return data


def check_binary_data_exists(data: Dict[str, pd.DataFrame]) -> bool:
    """Check if binary judge data exists for all policies."""
    for policy, df in data.items():
        if "binary_judge" not in df.columns or df["binary_judge"].isna().all():
            logger.warning(f"No binary judge data for {policy}")
            return False
    return True


def run_single_experiment(
    data: Dict[str, pd.DataFrame],
    sample_size: int,
    oracle_coverage: float,
    seed: int,
) -> Dict[str, Any]:
    """Run a single binary baseline experiment.

    Args:
        data: Dictionary mapping policy names to DataFrames
        sample_size: Number of rows to subsample
        oracle_coverage: Fraction of samples with oracle labels for calibration
        seed: Random seed

    Returns:
        Result dictionary with estimates for all policies
    """
    np.random.seed(seed)

    base_df = data["base"]
    n_base = len(base_df)

    # Subsample base policy
    if sample_size < n_base:
        base_indices = np.random.choice(n_base, sample_size, replace=False)
    else:
        base_indices = np.arange(n_base)

    # Split into calibration and test
    n_oracle = int(len(base_indices) * oracle_coverage)
    cal_indices = base_indices[:n_oracle]

    # Get calibration data from base
    y_cal = base_df.iloc[cal_indices]["oracle_label"].values
    z_cal = base_df.iloc[cal_indices]["binary_judge"].values

    # Filter out NaN binary judgments
    valid_cal = ~np.isnan(z_cal)
    y_cal = y_cal[valid_cal]
    z_cal = z_cal[valid_cal].astype(int)

    if len(y_cal) < 10:
        return {
            "seed": seed,
            "sample_size": sample_size,
            "oracle_coverage": oracle_coverage,
            "error": "Insufficient calibration samples",
        }

    # Compute confusion stats from base calibration set
    base_confusion = estimate_confusion_stats(y_cal, z_cal)

    results = {
        "seed": seed,
        "sample_size": sample_size,
        "oracle_coverage": oracle_coverage,
        "n_oracle": len(y_cal),
        "base_sensitivity": base_confusion.sensitivity,
        "base_specificity": base_confusion.specificity,
        "base_youdens_j": base_confusion.youdens_j,
        "policies": {},
    }

    # Evaluate each target policy
    for policy in TARGET_POLICIES:
        policy_df = data[policy]

        # Use same prompt subset as base
        prompt_ids = base_df.iloc[base_indices]["prompt_id"].values
        policy_subset = policy_df[policy_df["prompt_id"].isin(prompt_ids)]

        if len(policy_subset) == 0:
            results["policies"][policy] = {"error": "No matching prompts"}
            continue

        # Get binary predictions and oracle labels
        z_test = policy_subset["binary_judge"].values
        y_test = policy_subset["oracle_label"].values

        # Filter NaN
        valid_test = ~np.isnan(z_test)
        z_test = z_test[valid_test].astype(int)
        y_test = y_test[valid_test]
        prompt_ids_test = policy_subset[valid_test]["prompt_id"].values

        # Ground truth oracle mean
        oracle_mean = y_test.mean()

        # Binary naive estimate
        naive_est = binary_naive_estimate(z_test, cluster_ids=prompt_ids_test)

        # Binary adjusted estimate
        adjusted_est = binary_adjusted_estimate(z_test, base_confusion, cluster_ids=prompt_ids_test)

        # Compute transport audit if we have oracle labels for this policy
        # (We do for MBPP since we have full oracle coverage)
        policy_confusion = estimate_confusion_stats(y_test, z_test)
        transport_audit = confusion_matrix_transport_audit(base_confusion, policy_confusion)

        results["policies"][policy] = {
            "oracle_mean": oracle_mean,
            "binary_naive": naive_est.estimate,
            "binary_naive_se": naive_est.se,
            "binary_adjusted": adjusted_est.estimate,
            "binary_adjusted_se": adjusted_est.se,
            "adjusted_valid": adjusted_est.valid,
            "adjusted_diagnostics": adjusted_est.diagnostics,
            # Transport audit
            "transport_passes": transport_audit["passes_audit"],
            "transport_p_combined": transport_audit["p_combined"],
            "policy_sensitivity": policy_confusion.sensitivity,
            "policy_specificity": policy_confusion.specificity,
            "delta_se": transport_audit["delta_se"],
            "delta_sp": transport_audit["delta_sp"],
            # Errors
            "naive_error": naive_est.estimate - oracle_mean,
            "adjusted_error": (
                adjusted_est.estimate - oracle_mean
                if adjusted_est.valid else np.nan
            ),
        }

    return results


def run_experiment_worker(args: tuple) -> Dict[str, Any]:
    """Worker function for parallel execution."""
    data_dict, sample_size, oracle_coverage, seed = args

    # Convert dict back to DataFrames
    data = {k: pd.DataFrame(v) for k, v in data_dict.items()}

    return run_single_experiment(data, sample_size, oracle_coverage, seed)


def run_all_experiments(
    data: Dict[str, pd.DataFrame],
    max_workers: int = 8,
    test_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Run all binary baseline experiments.

    Args:
        data: Dictionary mapping policy names to DataFrames
        max_workers: Number of parallel workers
        test_mode: If True, only run 1 seed

    Returns:
        List of result dictionaries
    """
    # Generate experiment specs
    seeds = SEEDS[:1] if test_mode else SEEDS
    experiments = [
        (sample_size, oracle_coverage, seed)
        for seed in seeds
        for sample_size in SAMPLE_SIZES
        for oracle_coverage in ORACLE_COVERAGES
    ]

    logger.info(f"Running {len(experiments)} binary baseline experiments")

    # Convert DataFrames to dicts for pickling
    data_dict = {k: v.to_dict() for k, v in data.items()}

    results = []

    with tqdm(total=len(experiments), desc="Binary Baseline") as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_experiment_worker,
                    (data_dict, ss, oc, seed)
                ): (ss, oc, seed)
                for ss, oc, seed in experiments
            }

            for future in as_completed(futures):
                ss, oc, seed = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed: ss={ss}, oc={oc}, seed={seed}: {e}")
                    results.append({
                        "sample_size": ss,
                        "oracle_coverage": oc,
                        "seed": seed,
                        "error": str(e),
                    })
                pbar.update(1)

    return results


def compute_summary_statistics(results: List[Dict]) -> Dict:
    """Compute summary statistics from experiment results."""
    # Filter successful results
    valid_results = [r for r in results if "policies" in r]

    if not valid_results:
        return {"error": "No valid results"}

    # Aggregate by policy
    policy_stats = {}

    for policy in TARGET_POLICIES:
        naive_errors = []
        adjusted_errors = []
        transport_passes = []

        for r in valid_results:
            if policy in r["policies"]:
                p = r["policies"][policy]
                if "naive_error" in p and not np.isnan(p["naive_error"]):
                    naive_errors.append(p["naive_error"])
                if "adjusted_error" in p and not np.isnan(p["adjusted_error"]):
                    adjusted_errors.append(p["adjusted_error"])
                if "transport_passes" in p:
                    transport_passes.append(p["transport_passes"])

        policy_stats[policy] = {
            "n_experiments": len(naive_errors),
            "naive_bias": np.mean(naive_errors) if naive_errors else np.nan,
            "naive_rmse": np.sqrt(np.mean(np.array(naive_errors)**2)) if naive_errors else np.nan,
            "adjusted_bias": np.mean(adjusted_errors) if adjusted_errors else np.nan,
            "adjusted_rmse": np.sqrt(np.mean(np.array(adjusted_errors)**2)) if adjusted_errors else np.nan,
            "transport_pass_rate": np.mean(transport_passes) if transport_passes else np.nan,
        }

    # Overall transport-valid vs invalid
    transport_valid_policies = [p for p in TARGET_POLICIES
                                 if policy_stats[p]["transport_pass_rate"] > 0.5]
    transport_invalid_policies = [p for p in TARGET_POLICIES
                                   if policy_stats[p]["transport_pass_rate"] <= 0.5]

    return {
        "policy_stats": policy_stats,
        "transport_valid_policies": transport_valid_policies,
        "transport_invalid_policies": transport_invalid_policies,
        "n_experiments": len(valid_results),
    }


def save_results(results: List[Dict], output_file: Path) -> None:
    """Save results to JSONL file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(convert_numpy(result)) + "\n")

    logger.info(f"Saved {len(results)} results to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run binary baseline experiments")
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: run with 1 seed only",
    )
    parser.add_argument(
        "--check-data", action="store_true",
        help="Check if binary judge data exists and exit",
    )
    args = parser.parse_args()

    # Load data
    logger.info("Loading policy data...")
    data = load_all_policies()

    if not data:
        logger.error("Failed to load any policy data")
        return 1

    # Check for binary judge data
    has_binary = check_binary_data_exists(data)

    if args.check_data:
        if has_binary:
            print("✓ Binary judge data exists for all policies")
            for policy, df in data.items():
                yes_rate = df["binary_judge"].mean()
                print(f"  {policy}: {yes_rate:.1%} YES rate")
            return 0
        else:
            print("✗ Binary judge data missing")
            print("  Run: python data_generation_mbpp/add_binary_judge.py")
            return 1

    if not has_binary:
        logger.error("Binary judge data not found. Run add_binary_judge.py first.")
        return 1

    # Run experiments
    results = run_all_experiments(
        data,
        max_workers=args.workers,
        test_mode=args.test,
    )

    # Save results
    save_results(results, RESULTS_FILE)

    # Compute and display summary
    summary = compute_summary_statistics(results)

    print("\n" + "=" * 60)
    print("BINARY BASELINE SUMMARY")
    print("=" * 60)

    print(f"\nTotal experiments: {summary['n_experiments']}")
    print(f"Transport-valid policies: {summary['transport_valid_policies']}")
    print(f"Transport-invalid policies: {summary['transport_invalid_policies']}")

    print("\nPer-Policy Statistics:")
    for policy, stats in summary["policy_stats"].items():
        print(f"\n  {policy}:")
        print(f"    Naive RMSE:      {stats['naive_rmse']:.4f}")
        print(f"    Adjusted RMSE:   {stats['adjusted_rmse']:.4f}")
        print(f"    Transport pass:  {stats['transport_pass_rate']:.1%}")

    # Save summary
    summary_file = RESULTS_DIR / "binary_baseline_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to: {summary_file}")

    return 0


if __name__ == "__main__":
    exit(main())
