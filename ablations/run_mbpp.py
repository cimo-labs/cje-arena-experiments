#!/usr/bin/env python3
"""
MBPP ablation runner - Direct mode only.

This runs the MBPP coding benchmark ablation using Direct estimators
to test calibration effect on coding tasks with objective oracle (unit tests).

Usage:
    # Run all experiments in parallel
    python run_mbpp.py

    # Run sequentially for debugging
    python run_mbpp.py --sequential

    # Limit workers
    python run_mbpp.py --workers 4

    # Show experiment count only
    python run_mbpp.py --count
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add parent directories to path
import sys

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import MBPP-specific configuration
from config_mbpp import (
    EXPERIMENTS_MBPP,
    CONSTRAINTS_MBPP,
    DR_CONFIG_MBPP,
    DATA_PATH_MBPP,
    RESULTS_PATH_MBPP,
    CHECKPOINT_PATH_MBPP,
    RESULTS_DIR_MBPP,
    TARGET_POLICIES_MBPP,
)

# Import existing ablation infrastructure
from core.base import BaseAblation
from core.schemas import ExperimentSpec

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def run_mbpp_experiment_worker(
    spec_dict: Dict[str, Any], seed: int
) -> Dict[str, Any]:
    """Worker function for parallel execution of MBPP experiments.

    This function is defined at module level to be pickleable for ProcessPoolExecutor.

    Args:
        spec_dict: Serialized ExperimentSpec
        seed: Random seed for this experiment

    Returns:
        Result dictionary
    """
    # Import inside worker to avoid pickling issues
    from core.base import BaseAblation
    from core.schemas import ExperimentSpec

    # Set seeds for reproducibility in worker process
    np.random.seed(seed)
    random.seed(seed)

    # Create a new ablation instance
    ablation = MBPPAblation()

    # Reconstruct spec from dict
    spec = ExperimentSpec(**spec_dict)

    try:
        result = ablation.run_single(spec, seed)
        return result  # type: ignore[no-any-return]
    except Exception as e:
        logger.error(f"Failed to run {spec.estimator}: {e}")
        return {
            "spec": spec.to_dict(),
            "success": False,
            "error": str(e),
        }


class MBPPAblation(BaseAblation):
    """MBPP ablation for Direct estimators only."""

    def __init__(self) -> None:
        super().__init__(name="mbpp")
        self.checkpoint_file = (
            CHECKPOINT_PATH_MBPP
            if isinstance(CHECKPOINT_PATH_MBPP, Path)
            else Path(CHECKPOINT_PATH_MBPP)
        )
        self.results_file = (
            RESULTS_PATH_MBPP
            if isinstance(RESULTS_PATH_MBPP, Path)
            else Path(RESULTS_PATH_MBPP)
        )

        # Ensure results directory exists
        RESULTS_DIR_MBPP.mkdir(parents=True, exist_ok=True)

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
            return {k: MBPPAblation._convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [MBPPAblation._convert_numpy(v) for v in obj]
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
        """Load oracle ground truth values for MBPP (unit test pass rates).

        For MBPP, oracle_label is binary (0/1) indicating test pass/fail.
        Ground truth is the mean pass rate per policy.

        Args:
            dataset_path: Path to dataset file
            dataset: Dataset object
            target_policies: List of target policies

        Returns:
            Dictionary mapping policy names to oracle mean values (pass rates)
        """
        oracle_means = {}
        data_dir = Path(dataset_path).parent
        responses_dir = data_dir / "responses"

        # Load base policy oracle labels from dataset
        base_oracle_values = []
        for sample in dataset.samples:
            if hasattr(sample, "oracle_label") and sample.oracle_label is not None:
                base_oracle_values.append(sample.oracle_label)

        if base_oracle_values:
            oracle_means["base"] = float(np.mean(base_oracle_values))

        # Load oracle labels for each target policy from response files
        for policy in target_policies:
            response_file = responses_dir / f"{policy}_responses.jsonl"
            if response_file.exists():
                oracle_values = []
                with open(response_file, "r") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            # MBPP format: oracle_label in metadata
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
        """Run the MBPP ablation experiments.

        Args:
            parallel: Whether to run experiments in parallel
            max_workers: Max number of parallel workers (defaults to 8)

        Returns:
            List of result dictionaries
        """
        if parallel:
            return self._run_ablation_parallel(max_workers)
        else:
            return self._run_ablation_sequential()

    def _generate_all_specs(self) -> List[Tuple[ExperimentSpec, int]]:
        """Generate all experiment specifications for MBPP.

        Returns:
            List of (spec, seed) tuples
        """
        all_specs = []

        # SEEDS AS OUTERMOST LOOP
        for seed in EXPERIMENTS_MBPP["seeds"]:
            for estimator in EXPERIMENTS_MBPP["estimators"]:
                for sample_size in EXPERIMENTS_MBPP["sample_sizes"]:
                    for oracle_coverage in EXPERIMENTS_MBPP["oracle_coverages"]:
                        # Apply covariate constraints
                        if estimator in CONSTRAINTS_MBPP.get("never_covariates", set()):
                            covariate_options = [False]
                        else:
                            covariate_options = EXPERIMENTS_MBPP["use_covariates"]

                        for use_covariates in covariate_options:
                            spec = ExperimentSpec(
                                ablation="mbpp",
                                dataset_path=str(DATA_PATH_MBPP),
                                estimator=estimator,
                                sample_size=sample_size,
                                oracle_coverage=oracle_coverage,
                                rho=EXPERIMENTS_MBPP["var_cap"],
                                n_seeds=1,
                                seed_base=seed,
                                extra={
                                    "use_weight_calibration": False,  # Direct mode doesn't use weights
                                    "reward_calibration_mode": EXPERIMENTS_MBPP[
                                        "reward_calibration_mode"
                                    ],
                                    "use_covariates": use_covariates,
                                    "var_cap": EXPERIMENTS_MBPP["var_cap"],
                                    "target_policies": TARGET_POLICIES_MBPP,
                                    # Inference method for Direct estimator
                                    "inference_method": EXPERIMENTS_MBPP.get("inference_method", "cluster_robust"),
                                    "n_bootstrap": EXPERIMENTS_MBPP.get("n_bootstrap", 500),
                                },
                            )

                            # Check if already completed
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
        logger.info("MBPP ABLATION - Direct Mode Only")
        logger.info("=" * 60)
        logger.info(f"Parallel execution with {max_workers} workers")
        logger.info(f"Total experiments to run: {len(all_specs)}")
        logger.info(f"Already completed (cached): {len(self.completed_experiments)}")
        logger.info("=" * 60)

        with tqdm(total=len(all_specs), desc="MBPP Experiments") as pbar:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_spec = {
                    executor.submit(
                        run_mbpp_experiment_worker, spec.to_dict(), seed
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
        logger.info("MBPP ABLATION COMPLETE")
        logger.info(f"Completed: {completed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Results saved to: {self.results_file}")
        logger.info("=" * 60)

        return all_results

    def _run_ablation_sequential(self) -> List[Dict[str, Any]]:
        """Run ablation experiments sequentially."""
        all_results = []
        total_experiments = 0
        skipped = 0
        completed = 0
        failed = 0

        for seed in EXPERIMENTS_MBPP["seeds"]:
            logger.info(f"\n{'='*60}")
            logger.info(f"SEED {seed}")
            logger.info(f"{'='*60}")

            for estimator in EXPERIMENTS_MBPP["estimators"]:
                for sample_size in EXPERIMENTS_MBPP["sample_sizes"]:
                    for oracle_coverage in EXPERIMENTS_MBPP["oracle_coverages"]:
                        if estimator in CONSTRAINTS_MBPP.get("never_covariates", set()):
                            covariate_options = [False]
                        else:
                            covariate_options = EXPERIMENTS_MBPP["use_covariates"]

                        for use_covariates in covariate_options:
                            spec = ExperimentSpec(
                                ablation="mbpp",
                                dataset_path=str(DATA_PATH_MBPP),
                                estimator=estimator,
                                sample_size=sample_size,
                                oracle_coverage=oracle_coverage,
                                rho=EXPERIMENTS_MBPP["var_cap"],
                                n_seeds=1,
                                seed_base=seed,
                                extra={
                                    "use_weight_calibration": False,
                                    "reward_calibration_mode": EXPERIMENTS_MBPP[
                                        "reward_calibration_mode"
                                    ],
                                    "use_covariates": use_covariates,
                                    "var_cap": EXPERIMENTS_MBPP["var_cap"],
                                    "target_policies": TARGET_POLICIES_MBPP,
                                    # Inference method for Direct estimator
                                    "inference_method": EXPERIMENTS_MBPP.get("inference_method", "cluster_robust"),
                                    "n_bootstrap": EXPERIMENTS_MBPP.get("n_bootstrap", 500),
                                },
                            )

                            exp_id = self._generate_exp_id(spec.to_dict())
                            if exp_id in self.completed_experiments:
                                skipped += 1
                                continue

                            total_experiments += 1

                            cov_label = "+cov" if use_covariates else ""
                            logger.info(
                                f"\n[{completed + failed + 1}] "
                                f"{estimator}{cov_label} n={sample_size} "
                                f"oracle={oracle_coverage:.0%}"
                            )

                            try:
                                result = self.run_single(spec, seed)
                                self._save_checkpoint(result)
                                all_results.append(result)
                                completed += 1

                                if result.get("success"):
                                    rmse = result.get("rmse_vs_oracle", "N/A")
                                    if rmse != "N/A":
                                        logger.info(f"  ✓ RMSE: {rmse:.4f}")
                                    else:
                                        logger.info("  ✓ Completed")

                            except Exception as e:
                                logger.error(f"  ✗ Failed: {e}")
                                failed += 1
                                failed_result = {
                                    "spec": spec.to_dict(),
                                    "success": False,
                                    "error": str(e),
                                }
                                self._save_checkpoint(failed_result)

        logger.info("\n" + "=" * 60)
        logger.info("MBPP ABLATION COMPLETE")
        logger.info(f"Completed: {completed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Skipped (cached): {skipped}")
        logger.info(f"Results saved to: {self.results_file}")
        logger.info("=" * 60)

        return all_results


def main() -> None:
    """Run MBPP ablation experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run MBPP ablation experiments")
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
    args = parser.parse_args()

    if args.count:
        from config_mbpp import get_experiment_count
        print(f"Total experiments: {get_experiment_count():,}")
        return

    ablation = MBPPAblation()
    results = ablation.run_ablation(
        parallel=not args.sequential, max_workers=args.workers
    )

    if results:
        summary_file = RESULTS_DIR_MBPP / "mbpp_summary.json"
        with open(summary_file, "w") as f:
            summary_data = MBPPAblation._convert_numpy(
                {
                    "timestamp": time.time(),
                    "n_experiments": len(results),
                    "results": results,
                }
            )
            json.dump(summary_data, f, indent=2)
        logger.info(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
