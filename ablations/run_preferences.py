#!/usr/bin/env python3
"""
Preferences ablation runner - CJE vs Rogan-Gladen comparison.

This runs the Chatbot Arena preference ablation comparing:
1. Continuous CJE (naive-direct, direct)
2. Binary Rogan-Gladen (binary-naive, binary-adjusted)

Usage:
    # Run continuous CJE experiments
    python run_preferences.py

    # Run binary baseline comparison
    python run_preferences.py --binary

    # Run both
    python run_preferences.py --all

    # Show experiment count
    python run_preferences.py --count
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import random

# Add parent directories to path
import sys

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import preference-specific configuration
from config_preferences import (
    EXPERIMENTS_PREFERENCES,
    CONSTRAINTS_PREFERENCES,
    DR_CONFIG_PREFERENCES,
    DATA_PATH_PREFERENCES,
    RESULTS_PATH_PREFERENCES,
    CHECKPOINT_PATH_PREFERENCES,
    RESULTS_DIR_PREFERENCES,
    RESPONSES_DIR_PREFERENCES,
    TARGET_POLICIES_PREFERENCES,
    ALL_POLICIES_PREFERENCES,
    BINARY_RESULTS_PATH,
    BINARY_SUMMARY_PATH,
)

# Import existing ablation infrastructure
from core.base import BaseAblation
from core.schemas import ExperimentSpec

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def run_preferences_experiment_worker(
    spec_dict: Dict[str, Any], seed: int
) -> Dict[str, Any]:
    """Worker function for parallel execution."""
    from core.base import BaseAblation
    from core.schemas import ExperimentSpec

    # Set seeds for reproducibility in worker process
    np.random.seed(seed)
    random.seed(seed)

    ablation = PreferencesAblation()
    spec = ExperimentSpec(**spec_dict)

    try:
        result = ablation.run_single(spec, seed)
        return result
    except Exception as e:
        logger.error(f"Failed to run {spec.estimator}: {e}")
        return {
            "spec": spec.to_dict(),
            "success": False,
            "error": str(e),
        }


class PreferencesAblation(BaseAblation):
    """Preferences ablation for comparing CJE vs Rogan-Gladen."""

    def __init__(self) -> None:
        super().__init__(name="preferences")
        self.checkpoint_file = (
            CHECKPOINT_PATH_PREFERENCES
            if isinstance(CHECKPOINT_PATH_PREFERENCES, Path)
            else Path(CHECKPOINT_PATH_PREFERENCES)
        )
        self.results_file = (
            RESULTS_PATH_PREFERENCES
            if isinstance(RESULTS_PATH_PREFERENCES, Path)
            else Path(RESULTS_PATH_PREFERENCES)
        )

        # Ensure results directory exists
        RESULTS_DIR_PREFERENCES.mkdir(parents=True, exist_ok=True)

        self.completed_experiments = self._load_checkpoint()

    @staticmethod
    def _convert_numpy(obj: Any) -> Any:
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: PreferencesAblation._convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [PreferencesAblation._convert_numpy(v) for v in obj]
        elif hasattr(obj, "item"):
            return obj.item()
        return obj

    def _load_checkpoint(self) -> set:
        """Load completed experiment IDs from checkpoint."""
        completed = set()
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, "r") as f:
                for line in f:
                    try:
                        result = json.loads(line)
                        if result.get("success"):
                            exp_id = self._generate_exp_id(result.get("spec", {}))
                            completed.add(exp_id)
                    except json.JSONDecodeError:
                        continue
            logger.info(
                f"Loaded {len(completed)} completed experiments from checkpoint"
            )
        return completed

    def _generate_exp_id(self, spec: Dict[str, Any]) -> str:
        """Generate unique experiment ID from specification."""
        key_params = [
            spec.get("estimator", ""),
            str(spec.get("sample_size", "")),
            str(spec.get("oracle_coverage", "")),
            str(spec.get("extra", {}).get("use_covariates", False)),
            str(spec.get("seed_base", 42)),
        ]
        return "_".join(key_params)

    def _save_checkpoint(self, result: Dict[str, Any]) -> None:
        """Append result to checkpoint file."""
        result = self._convert_numpy(result)

        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    def _load_oracle_ground_truth(
        self, dataset_path: str, dataset: Any, target_policies: List[str]
    ) -> Dict[str, float]:
        """Load oracle ground truth values for preferences.

        For preferences, oracle_label is binary (0/1) indicating human preference.
        Ground truth is the mean "A wins" rate per policy.
        """
        oracle_means = {}
        data_dir = Path(dataset_path).parent
        responses_dir = data_dir / "responses"

        # Load oracle labels for each policy from response files
        for policy in target_policies:
            response_file = responses_dir / f"{policy}_responses.jsonl"
            if response_file.exists():
                oracle_values = []
                with open(response_file, "r") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if (
                                "metadata" in data
                                and "oracle_label" in data["metadata"]
                            ):
                                oracle_val = data["metadata"]["oracle_label"]
                                if oracle_val is not None:
                                    oracle_values.append(oracle_val)
                        except json.JSONDecodeError:
                            continue

                if oracle_values:
                    oracle_means[policy] = float(np.mean(oracle_values))

        return oracle_means

    def run_ablation(
        self, parallel: bool = True, max_workers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Run the preferences ablation experiments."""
        if parallel:
            return self._run_ablation_parallel(max_workers)
        else:
            return self._run_ablation_sequential()

    def _generate_all_specs(self) -> List[Tuple[ExperimentSpec, int]]:
        """Generate all experiment specifications for preferences."""
        all_specs = []

        for seed in EXPERIMENTS_PREFERENCES["seeds"]:
            for estimator in EXPERIMENTS_PREFERENCES["estimators"]:
                for sample_size in EXPERIMENTS_PREFERENCES["sample_sizes"]:
                    for oracle_coverage in EXPERIMENTS_PREFERENCES["oracle_coverages"]:
                        if estimator in CONSTRAINTS_PREFERENCES.get("never_covariates", set()):
                            covariate_options = [False]
                        else:
                            covariate_options = EXPERIMENTS_PREFERENCES["use_covariates"]

                        for use_covariates in covariate_options:
                            spec = ExperimentSpec(
                                ablation="preferences",
                                dataset_path=str(DATA_PATH_PREFERENCES),
                                estimator=estimator,
                                sample_size=sample_size,
                                oracle_coverage=oracle_coverage,
                                rho=EXPERIMENTS_PREFERENCES["var_cap"],
                                n_seeds=1,
                                seed_base=seed,
                                extra={
                                    "use_weight_calibration": False,
                                    "reward_calibration_mode": EXPERIMENTS_PREFERENCES[
                                        "reward_calibration_mode"
                                    ],
                                    "use_covariates": use_covariates,
                                    "var_cap": EXPERIMENTS_PREFERENCES["var_cap"],
                                    "target_policies": TARGET_POLICIES_PREFERENCES,
                                    "inference_method": EXPERIMENTS_PREFERENCES.get("inference_method", "cluster_robust"),
                                    "n_bootstrap": EXPERIMENTS_PREFERENCES.get("n_bootstrap", 500),
                                },
                            )

                            exp_id = self._generate_exp_id(spec.to_dict())
                            if exp_id not in self.completed_experiments:
                                all_specs.append((spec, seed))

        return all_specs

    def _run_ablation_parallel(
        self, max_workers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Run ablation experiments in parallel."""
        all_specs = self._generate_all_specs()

        if not all_specs:
            logger.info("All experiments already completed!")
            return []

        if max_workers is None:
            max_workers = 8

        all_results = []
        completed = 0
        failed = 0

        logger.info("=" * 60)
        logger.info("PREFERENCES ABLATION - CJE Continuous Methods")
        logger.info("=" * 60)
        logger.info(f"Parallel execution with {max_workers} workers")
        logger.info(f"Total experiments to run: {len(all_specs)}")
        logger.info(f"Already completed (cached): {len(self.completed_experiments)}")
        logger.info("=" * 60)

        with tqdm(total=len(all_specs), desc="Preferences Experiments") as pbar:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_spec = {
                    executor.submit(
                        run_preferences_experiment_worker, spec.to_dict(), seed
                    ): (spec, seed)
                    for spec, seed in all_specs
                }

                for future in as_completed(future_to_spec):
                    spec, seed = future_to_spec[future]

                    try:
                        result = future.result()
                        self._save_checkpoint(result)
                        all_results.append(result)

                        if result.get("success"):
                            completed += 1
                            rmse = result.get("rmse_vs_oracle", "N/A")
                            pbar.set_postfix(
                                completed=completed,
                                failed=failed,
                                last=(
                                    f"{spec.estimator} RMSE={rmse:.4f}"
                                    if rmse != "N/A"
                                    else spec.estimator
                                ),
                            )
                        else:
                            failed += 1
                            logger.error(
                                f"Failed: {spec.estimator} - {result.get('error', 'Unknown error')}"
                            )

                    except Exception as e:
                        failed += 1
                        logger.error(f"Exception running {spec.estimator}: {e}")

                        failed_result = {
                            "spec": spec.to_dict(),
                            "success": False,
                            "error": str(e),
                        }
                        self._save_checkpoint(failed_result)
                        all_results.append(failed_result)

                    pbar.update(1)

        logger.info("\n" + "=" * 60)
        logger.info("PREFERENCES ABLATION COMPLETE")
        logger.info(f"Completed: {completed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Results saved to: {self.results_file}")
        logger.info("=" * 60)

        return all_results

    def _run_ablation_sequential(self) -> List[Dict[str, Any]]:
        """Run ablation experiments sequentially."""
        all_specs = self._generate_all_specs()
        all_results = []
        completed = 0
        failed = 0

        for spec, seed in tqdm(all_specs, desc="Preferences Experiments"):
            try:
                result = self.run_single(spec, seed)
                self._save_checkpoint(result)
                all_results.append(result)

                if result.get("success"):
                    completed += 1
                else:
                    failed += 1

            except Exception as e:
                failed += 1
                logger.error(f"Failed: {spec.estimator} - {e}")

        logger.info(f"Completed: {completed}, Failed: {failed}")
        return all_results


def run_binary_baseline() -> List[Dict[str, Any]]:
    """Run binary baseline experiments using Rogan-Gladen adjustment.

    This implements the binary comparison from the MBPP analysis:
    1. Load binary judge scores (A/B predictions)
    2. Compute confusion matrix (Se, Sp) on oracle-labeled subset
    3. Apply Rogan-Gladen adjustment: θ̂ = (p̂ + Sp - 1) / J

    Returns:
        List of experiment results
    """
    logger.info("=" * 60)
    logger.info("BINARY BASELINE - Rogan-Gladen Adjustment")
    logger.info("=" * 60)

    results = []
    completed_ids = set()

    # Load existing results
    if BINARY_RESULTS_PATH.exists():
        with open(BINARY_RESULTS_PATH, 'r') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    if result.get("success"):
                        exp_id = f"{result['sample_size']}_{result['oracle_coverage']}_{result['seed']}"
                        completed_ids.add(exp_id)
                except json.JSONDecodeError:
                    continue
        logger.info(f"Loaded {len(completed_ids)} completed binary experiments")

    # Load policy response files (all policies for binary baseline)
    policy_data = {}
    for policy in ALL_POLICIES_PREFERENCES:
        response_file = RESPONSES_DIR_PREFERENCES / f"{policy}_responses.jsonl"
        if not response_file.exists():
            logger.warning(f"Missing response file: {response_file}")
            continue

        records = []
        with open(response_file, 'r') as f:
            for line in f:
                records.append(json.loads(line))
        policy_data[policy] = records
        logger.info(f"Loaded {len(records)} records for {policy}")

    if not policy_data:
        logger.error("No policy data found. Run add_binary_judge.py first.")
        return []

    # Experiment grid
    total_experiments = (
        len(EXPERIMENTS_PREFERENCES["sample_sizes"])
        * len(EXPERIMENTS_PREFERENCES["oracle_coverages"])
        * len(EXPERIMENTS_PREFERENCES["seeds"])
    )

    with tqdm(total=total_experiments, desc="Binary experiments") as pbar:
        for seed in EXPERIMENTS_PREFERENCES["seeds"]:
            for sample_size in EXPERIMENTS_PREFERENCES["sample_sizes"]:
                for oracle_coverage in EXPERIMENTS_PREFERENCES["oracle_coverages"]:
                    exp_id = f"{sample_size}_{oracle_coverage}_{seed}"
                    if exp_id in completed_ids:
                        pbar.update(1)
                        continue

                    result = run_single_binary_experiment(
                        policy_data,
                        sample_size=sample_size,
                        oracle_coverage=oracle_coverage,
                        seed=int(seed),
                    )

                    # Save result
                    BINARY_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with open(BINARY_RESULTS_PATH, 'a') as f:
                        f.write(json.dumps(result) + '\n')

                    results.append(result)
                    pbar.update(1)

    # Aggregate and save summary
    if results:
        summary = aggregate_binary_results(results)
        with open(BINARY_SUMMARY_PATH, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {BINARY_SUMMARY_PATH}")

    return results


def run_single_binary_experiment(
    policy_data: Dict[str, List[Dict]],
    sample_size: int,
    oracle_coverage: float,
    seed: int,
) -> Dict[str, Any]:
    """Run a single binary baseline experiment."""
    from binary_baseline import lee_et_al_confidence_interval

    random.seed(seed)
    np.random.seed(seed)

    result = {
        "sample_size": sample_size,
        "oracle_coverage": oracle_coverage,
        "seed": seed,
        "success": False,
        "estimates_naive": {},
        "estimates_adjusted": {},
        "oracle_truths": {},
        "confusion_matrix": {},
        "ci_lower": {},
        "ci_upper": {},
        "ci_covers_oracle": {},
    }

    try:
        for policy, records in policy_data.items():
            # Subsample
            n_samples = min(sample_size, len(records))
            indices = sorted(random.sample(range(len(records)), n_samples))
            samples = [records[i] for i in indices]

            # Split into oracle-labeled and unlabeled
            oracle_indices = []
            for i, s in enumerate(samples):
                if s.get('metadata', {}).get('oracle_label') is not None:
                    oracle_indices.append(i)

            n_oracle = max(2, int(len(oracle_indices) * oracle_coverage))
            oracle_subset = random.sample(oracle_indices, min(n_oracle, len(oracle_indices)))
            oracle_set = set(oracle_subset)

            # Extract binary predictions and oracle labels
            binary_preds = []
            oracle_labels = []
            for i, s in enumerate(samples):
                binary_judge = s.get('metadata', {}).get('binary_judge')
                oracle_label = s.get('metadata', {}).get('oracle_label')

                if binary_judge is not None:
                    binary_preds.append(binary_judge)
                    if i in oracle_set and oracle_label is not None:
                        oracle_labels.append((binary_judge, oracle_label))

            if not binary_preds or len(oracle_labels) < 5:
                continue

            # Compute naive estimate (proportion saying A)
            p_hat = np.mean(binary_preds)
            result["estimates_naive"][policy] = float(p_hat)

            # Compute confusion matrix on oracle subset
            tp = sum(1 for b, o in oracle_labels if b == 1 and o == 1)
            tn = sum(1 for b, o in oracle_labels if b == 0 and o == 0)
            fp = sum(1 for b, o in oracle_labels if b == 1 and o == 0)
            fn = sum(1 for b, o in oracle_labels if b == 0 and o == 1)

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.5
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.5
            youden_j = sensitivity + specificity - 1

            result["confusion_matrix"][policy] = {
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "youden_j": youden_j,
            }

            # Rogan-Gladen adjustment
            if youden_j > 0.01:  # Avoid division by near-zero
                theta_hat = (p_hat + specificity - 1) / youden_j
                theta_hat = max(0, min(1, theta_hat))  # Clip to [0, 1]
            else:
                theta_hat = p_hat  # Fall back to naive if J ≈ 0

            result["estimates_adjusted"][policy] = float(theta_hat)

            # Compute Lee et al. (2026) confidence interval
            n_test = len(binary_preds)
            n_pos_cal = tp + fn  # calibration positives
            n_neg_cal = tn + fp  # calibration negatives
            ci_lower, ci_upper = lee_et_al_confidence_interval(
                p=p_hat,
                q0=specificity,
                q1=sensitivity,
                n=n_test,
                m0=n_neg_cal,
                m1=n_pos_cal,
            )
            result["ci_lower"][policy] = float(ci_lower) if not np.isnan(ci_lower) else None
            result["ci_upper"][policy] = float(ci_upper) if not np.isnan(ci_upper) else None

            # Oracle ground truth (full sample)
            all_oracle = [
                r.get('metadata', {}).get('oracle_label')
                for r in records
                if r.get('metadata', {}).get('oracle_label') is not None
            ]
            if all_oracle:
                oracle_truth = float(np.mean(all_oracle))
                result["oracle_truths"][policy] = oracle_truth

                # Check if CI covers oracle truth
                if ci_lower is not None and ci_upper is not None and not np.isnan(ci_lower) and not np.isnan(ci_upper):
                    covers = ci_lower <= oracle_truth <= ci_upper
                    result["ci_covers_oracle"][policy] = bool(covers)  # Convert numpy bool to Python bool
                else:
                    result["ci_covers_oracle"][policy] = None

        # Compute RMSE
        if result["estimates_naive"] and result["oracle_truths"]:
            naive_errors = [
                (result["estimates_naive"].get(p, 0) - result["oracle_truths"].get(p, 0)) ** 2
                for p in result["oracle_truths"]
                if p in result["estimates_naive"]
            ]
            adjusted_errors = [
                (result["estimates_adjusted"].get(p, 0) - result["oracle_truths"].get(p, 0)) ** 2
                for p in result["oracle_truths"]
                if p in result["estimates_adjusted"]
            ]

            if naive_errors:
                result["rmse_naive"] = float(np.sqrt(np.mean(naive_errors)))
            if adjusted_errors:
                result["rmse_adjusted"] = float(np.sqrt(np.mean(adjusted_errors)))

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def aggregate_binary_results(results: List[Dict]) -> Dict[str, Any]:
    """Aggregate binary experiment results."""
    successful = [r for r in results if r.get("success")]

    if not successful:
        return {"n_experiments": len(results), "all_failed": True}

    summary = {
        "n_experiments": len(successful),
        "policy_stats": {},
        "overall_stats": {},
    }

    # Aggregate by policy
    policies = set()
    for r in successful:
        policies.update(r.get("estimates_naive", {}).keys())

    all_coverage = []  # Track all coverage results across policies

    for policy in policies:
        naive_errors = []
        adjusted_errors = []
        youden_js = []
        coverage_results = []

        for r in successful:
            oracle = r.get("oracle_truths", {}).get(policy)
            naive = r.get("estimates_naive", {}).get(policy)
            adjusted = r.get("estimates_adjusted", {}).get(policy)
            cm = r.get("confusion_matrix", {}).get(policy, {})
            covers = r.get("ci_covers_oracle", {}).get(policy)

            if oracle is not None and naive is not None:
                naive_errors.append((naive - oracle) ** 2)
            if oracle is not None and adjusted is not None:
                adjusted_errors.append((adjusted - oracle) ** 2)
            if "youden_j" in cm:
                youden_js.append(cm["youden_j"])
            if covers is not None:
                coverage_results.append(covers)
                all_coverage.append(covers)

        if naive_errors:
            policy_stats = {
                "naive_rmse": float(np.sqrt(np.mean(naive_errors))),
                "naive_bias": float(np.mean([np.sqrt(e) * np.sign(r.get("estimates_naive", {}).get(policy, 0) - r.get("oracle_truths", {}).get(policy, 0)) for e, r in zip(naive_errors, successful) if r.get("oracle_truths", {}).get(policy) is not None])),
                "adjusted_rmse": float(np.sqrt(np.mean(adjusted_errors))) if adjusted_errors else None,
                "mean_youden_j": float(np.mean(youden_js)) if youden_js else None,
            }
            # Add coverage
            if coverage_results:
                policy_stats["coverage_95ci"] = float(np.mean(coverage_results))
                policy_stats["n_coverage"] = len(coverage_results)
            summary["policy_stats"][policy] = policy_stats

    # Overall coverage across all policies
    if all_coverage:
        summary["overall_stats"]["coverage_95ci"] = float(np.mean(all_coverage))
        summary["overall_stats"]["n_coverage"] = len(all_coverage)

    return summary


def run_threshold_sweep(
    thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
    sample_sizes: List[int] = [5000],
    oracle_coverages: List[float] = [0.05],
    seeds: List[int] = list(range(30)),
    policies: List[str] = None,
) -> Dict[str, Any]:
    """Run Rogan-Gladen analysis across different thresholds.

    Instead of using native binary_judge, this thresholds the continuous
    judge_score at various thresholds to test threshold sensitivity.

    Args:
        thresholds: Score thresholds to test (0-1 scale)
        sample_sizes: Sample sizes to test
        oracle_coverages: Oracle coverage fractions to test
        seeds: Random seeds for replication
        policies: Policies to evaluate (default: target policies)

    Returns:
        Dictionary with results per threshold
    """
    from binary_baseline import lee_et_al_confidence_interval, get_thresholded_binary

    if policies is None:
        policies = TARGET_POLICIES_PREFERENCES

    # Load data for all policies
    policy_data = {}
    for policy in policies:
        response_file = RESPONSES_DIR_PREFERENCES / f"{policy}_responses.jsonl"
        if not response_file.exists():
            logger.warning(f"Missing response file: {response_file}")
            continue

        records = []
        with open(response_file, 'r') as f:
            for line in f:
                records.append(json.loads(line))
        policy_data[policy] = records
        logger.info(f"Loaded {len(records)} records for {policy}")

    if not policy_data:
        logger.error("No policy data found.")
        return {}

    results_by_threshold = {}

    for threshold in thresholds:
        logger.info(f"\n=== Running threshold {threshold} ===")
        threshold_results = []

        for seed in tqdm(seeds, desc=f"Threshold {threshold}"):
            for sample_size in sample_sizes:
                for oracle_coverage in oracle_coverages:
                    random.seed(seed)
                    np.random.seed(seed)

                    result = {
                        "threshold": threshold,
                        "sample_size": sample_size,
                        "oracle_coverage": oracle_coverage,
                        "seed": seed,
                        "success": False,
                        "estimates_adjusted": {},
                        "oracle_truths": {},
                        "confusion_matrix": {},
                        "ci_lower": {},
                        "ci_upper": {},
                        "ci_covers_oracle": {},
                    }

                    try:
                        for policy, records in policy_data.items():
                            # Subsample
                            n_samples = min(sample_size, len(records))
                            indices = sorted(random.sample(range(len(records)), n_samples))
                            samples = [records[i] for i in indices]

                            # Get thresholded binary predictions
                            binary_preds = get_thresholded_binary(samples, threshold)

                            # Split into oracle-labeled and unlabeled
                            oracle_indices = []
                            for i, s in enumerate(samples):
                                if s.get('metadata', {}).get('oracle_label') is not None:
                                    oracle_indices.append(i)

                            n_oracle = max(2, int(len(oracle_indices) * oracle_coverage))
                            oracle_subset = random.sample(oracle_indices, min(n_oracle, len(oracle_indices)))
                            oracle_set = set(oracle_subset)

                            # Extract oracle labels for confusion matrix
                            oracle_labels = []
                            for i in oracle_set:
                                s = samples[i]
                                oracle_label = s.get('metadata', {}).get('oracle_label')
                                if oracle_label is not None and not np.isnan(binary_preds[i]):
                                    oracle_labels.append((int(binary_preds[i]), oracle_label))

                            if len(oracle_labels) < 5:
                                continue

                            # Remove NaN predictions
                            valid_preds = binary_preds[~np.isnan(binary_preds)]
                            if len(valid_preds) == 0:
                                continue

                            p_hat = np.mean(valid_preds)

                            # Compute confusion matrix on oracle subset
                            tp = sum(1 for b, o in oracle_labels if b == 1 and o == 1)
                            tn = sum(1 for b, o in oracle_labels if b == 0 and o == 0)
                            fp = sum(1 for b, o in oracle_labels if b == 1 and o == 0)
                            fn = sum(1 for b, o in oracle_labels if b == 0 and o == 1)

                            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.5
                            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.5
                            youden_j = sensitivity + specificity - 1

                            result["confusion_matrix"][policy] = {
                                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                                "sensitivity": sensitivity,
                                "specificity": specificity,
                                "youden_j": youden_j,
                            }

                            # Rogan-Gladen adjustment
                            if youden_j > 0.01:
                                theta_hat = (p_hat + specificity - 1) / youden_j
                                theta_hat = max(0, min(1, theta_hat))
                            else:
                                theta_hat = p_hat

                            result["estimates_adjusted"][policy] = float(theta_hat)

                            # Compute Lee et al. confidence interval
                            n_test = len(valid_preds)
                            n_pos_cal = tp + fn
                            n_neg_cal = tn + fp
                            ci_lower, ci_upper = lee_et_al_confidence_interval(
                                p=p_hat,
                                q0=specificity,
                                q1=sensitivity,
                                n=n_test,
                                m0=n_neg_cal,
                                m1=n_pos_cal,
                            )
                            result["ci_lower"][policy] = float(ci_lower) if not np.isnan(ci_lower) else None
                            result["ci_upper"][policy] = float(ci_upper) if not np.isnan(ci_upper) else None

                            # Oracle ground truth
                            all_oracle = [
                                r.get('metadata', {}).get('oracle_label')
                                for r in records
                                if r.get('metadata', {}).get('oracle_label') is not None
                            ]
                            if all_oracle:
                                oracle_truth = float(np.mean(all_oracle))
                                result["oracle_truths"][policy] = oracle_truth

                                if ci_lower is not None and ci_upper is not None:
                                    covers = ci_lower <= oracle_truth <= ci_upper
                                    result["ci_covers_oracle"][policy] = bool(covers)

                        result["success"] = True

                    except Exception as e:
                        result["error"] = str(e)

                    threshold_results.append(result)

        # Aggregate results for this threshold
        successful = [r for r in threshold_results if r["success"]]
        if successful:
            # Compute aggregate metrics
            all_youden_j = []
            all_rmse = []
            all_coverage = []
            all_half_width = []

            for r in successful:
                for policy in policies:
                    if policy in r.get("confusion_matrix", {}):
                        all_youden_j.append(r["confusion_matrix"][policy]["youden_j"])

                    oracle = r.get("oracle_truths", {}).get(policy)
                    adjusted = r.get("estimates_adjusted", {}).get(policy)
                    if oracle is not None and adjusted is not None:
                        all_rmse.append((adjusted - oracle) ** 2)

                    covers = r.get("ci_covers_oracle", {}).get(policy)
                    if covers is not None:
                        all_coverage.append(covers)

                    ci_l = r.get("ci_lower", {}).get(policy)
                    ci_u = r.get("ci_upper", {}).get(policy)
                    if ci_l is not None and ci_u is not None:
                        all_half_width.append((ci_u - ci_l) / 2)

            results_by_threshold[threshold] = {
                "n_experiments": len(successful),
                "mean_youden_j": float(np.mean(all_youden_j)) if all_youden_j else None,
                "rmse": float(np.sqrt(np.mean(all_rmse))) if all_rmse else None,
                "coverage": float(np.mean(all_coverage)) if all_coverage else None,
                "mean_half_width": float(np.mean(all_half_width)) if all_half_width else None,
                "raw_results": threshold_results,
            }

    # Save results
    output_path = RESULTS_DIR_PREFERENCES / "threshold_sweep.json"
    with open(output_path, 'w') as f:
        # Don't save raw results to keep file small
        summary = {
            k: {kk: vv for kk, vv in v.items() if kk != "raw_results"}
            for k, v in results_by_threshold.items()
        }
        json.dump(summary, f, indent=2)
    logger.info(f"Threshold sweep saved to {output_path}")

    # Print summary table
    print("\n=== Threshold Sweep Results ===")
    print(f"{'Threshold':>10} {'Youden J':>10} {'RMSE':>10} {'Coverage':>10} {'Half-Width':>12}")
    print("-" * 54)
    for t in sorted(results_by_threshold.keys()):
        r = results_by_threshold[t]
        print(f"{t:>10.2f} {r['mean_youden_j']:>10.3f} {r['rmse']:>10.4f} {r['coverage']:>10.1%} {r['mean_half_width']:>12.3f}")

    return results_by_threshold


def main() -> None:
    """Run preferences ablation experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run preferences ablation experiments")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run experiments sequentially instead of in parallel",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Show experiment count and exit",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Run binary baseline (Rogan-Gladen) only",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run both continuous and binary experiments",
    )
    parser.add_argument(
        "--threshold-sweep",
        action="store_true",
        help="Run threshold sweep for binary method (tests different threshold values)",
    )
    args = parser.parse_args()

    if args.count:
        from config_preferences import get_experiment_count, get_binary_experiment_count
        print(f"Continuous experiments: {get_experiment_count():,}")
        print(f"Binary experiments: {get_binary_experiment_count():,}")
        return

    if args.threshold_sweep:
        run_threshold_sweep()
        return

    if args.binary:
        run_binary_baseline()
        return

    # Run continuous CJE experiments
    ablation = PreferencesAblation()
    results = ablation.run_ablation(
        parallel=not args.sequential, max_workers=args.workers
    )

    if results:
        summary_file = RESULTS_DIR_PREFERENCES / "preferences_summary.json"
        with open(summary_file, "w") as f:
            summary_data = PreferencesAblation._convert_numpy(
                {
                    "timestamp": time.time(),
                    "n_experiments": len(results),
                    "results": results,
                }
            )
            json.dump(summary_data, f, indent=2)
        logger.info(f"Summary saved to: {summary_file}")

    # Also run binary baseline if --all
    if args.all:
        run_binary_baseline()


if __name__ == "__main__":
    main()
