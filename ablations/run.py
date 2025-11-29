#!/usr/bin/env python3
"""
Unified experiment runner using existing ablation infrastructure.

This leverages the BaseAblation class to run a complete parameter sweep
for the paper, testing all key design decisions systematically.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from itertools import product
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add ablations directory to path for local imports
import sys

sys.path.insert(0, str(Path(__file__).parent))

# Import configuration
from config import (
    EXPERIMENTS,
    CONSTRAINTS,
    DR_CONFIG,
    DATA_PATH,
    RESULTS_PATH,
    CHECKPOINT_PATH,
)

# Import existing ablation infrastructure
from core.base import BaseAblation
from core.schemas import ExperimentSpec, aggregate_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def run_single_experiment_worker(
    spec_dict: Dict[str, Any], seed: int
) -> Dict[str, Any]:
    """Worker function for parallel execution of experiments.

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

    # Create a new ablation instance
    ablation = UnifiedAblation()

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


class UnifiedAblation(BaseAblation):
    """Unified ablation that tests all parameters systematically."""

    def __init__(self) -> None:
        super().__init__(name="unified")
        self.checkpoint_file = (
            CHECKPOINT_PATH
            if isinstance(CHECKPOINT_PATH, Path)
            else Path(CHECKPOINT_PATH)
        )
        self.results_file = (
            RESULTS_PATH if isinstance(RESULTS_PATH, Path) else Path(RESULTS_PATH)
        )
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
            return {k: UnifiedAblation._convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [UnifiedAblation._convert_numpy(v) for v in obj]
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
                            # Create ID from parameters
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
        # Create a deterministic string from key parameters
        key_params = [
            spec.get("estimator", ""),
            str(spec.get("sample_size", "")),
            str(spec.get("oracle_coverage", "")),
            str(spec.get("extra", {}).get("use_weight_calibration", False)),
            str(spec.get("extra", {}).get("reward_calibration_mode", "monotone")),
            str(spec.get("extra", {}).get("use_covariates", False)),
            # CF-bits removed from library
            str(spec.get("seed_base", 42)),  # This now properly includes the seed
        ]
        return "_".join(key_params)

    def _save_checkpoint(self, result: Dict[str, Any]) -> None:
        """Append result to checkpoint file."""
        # Convert numpy types before saving
        result = self._convert_numpy(result)

        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        # Also save to main results file
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    def create_estimator(
        self, spec: ExperimentSpec, sampler: Any, cal_result: Any, seed: int = 42
    ) -> Any:
        """Create estimator with unified parameter handling.

        This extends the base class method to handle our new parameters.
        """
        estimator_name = spec.estimator
        use_weight_calibration = spec.extra.get("use_weight_calibration", False)

        # Handle DR methods with our parameters
        if estimator_name in ["dr-cpo", "tmle", "mrdr"]:
            from cje.estimators.dr_base import DRCPOEstimator
            from cje.estimators.tmle import TMLEEstimator
            from cje.estimators.mrdr import MRDREstimator

            estimator_class = {
                "dr-cpo": DRCPOEstimator,
                "tmle": TMLEEstimator,
                "mrdr": MRDREstimator,
            }[estimator_name]

            kwargs = {
                "sampler": sampler,
                "n_folds": DR_CONFIG["n_folds"],
                "use_calibrated_weights": use_weight_calibration,  # Controls SIMCal for weights
            }

            # Always pass reward calibrator for outcome model (if available)
            # This is independent of weight calibration
            if cal_result and cal_result.calibrator:
                kwargs["reward_calibrator"] = cal_result.calibrator

            return estimator_class(**kwargs)

        elif estimator_name == "stacked-dr":
            from cje.estimators.stacking import StackedDREstimator

            kwargs = {
                "sampler": sampler,
                # Use default estimators (dr-cpo, tmle, mrdr)
                "n_folds": DR_CONFIG["n_folds"],  # Inner folds for component estimators
                "V_folds": DR_CONFIG.get(
                    "v_folds_stacking", 20
                ),  # Use config value, default 20
                "parallel": True,  # Enable parallel execution of component estimators
                "covariance_regularization": 1e-4,  # Add regularization for numerical stability
                "use_calibrated_weights": use_weight_calibration,  # Controls SIMCal for weights
                "weight_shrinkage": 0.0,  # No shrinkage - let optimizer find optimal weights
            }

            # Always pass reward calibrator for outcome model (if available)
            # This is independent of weight calibration
            if cal_result and cal_result.calibrator:
                kwargs["reward_calibrator"] = cal_result.calibrator

            return StackedDREstimator(**kwargs)

        else:
            # Fall back to base implementation
            return super().create_estimator(spec, sampler, cal_result, seed)

    def run_single_experiment(
        self, spec_dict: Dict[str, Any], seed: int
    ) -> Dict[str, Any]:
        """Run a single experiment (pickleable for parallel execution).

        Args:
            spec_dict: Serialized ExperimentSpec
            seed: Random seed for this experiment

        Returns:
            Result dictionary
        """
        # Reconstruct spec from dict
        spec = ExperimentSpec(**spec_dict)

        try:
            result = self.run_single(spec, seed)
            return result  # type: ignore[no-any-return]
        except Exception as e:
            logger.error(f"Failed to run {spec.estimator}: {e}")
            return {
                "spec": spec.to_dict(),
                "success": False,
                "error": str(e),
            }

    def run_ablation(
        self, parallel: bool = True, max_workers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Run the complete unified ablation.

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
        """Generate all experiment specifications.

        Returns:
            List of (spec, seed) tuples
        """
        all_specs = []

        # SEEDS AS OUTERMOST LOOP - ensures all estimators are evaluated with seed 0 first
        for seed in EXPERIMENTS["seeds"]:
            for estimator in EXPERIMENTS["estimators"]:
                for sample_size in EXPERIMENTS["sample_sizes"]:
                    for oracle_coverage in EXPERIMENTS["oracle_coverages"]:

                        # Apply calibration constraints
                        if estimator in CONSTRAINTS.get("requires_calibration", set()):
                            calibration_options = [True]
                        elif estimator in CONSTRAINTS.get("never_calibrated", set()):
                            calibration_options = [False]
                        else:
                            calibration_options = EXPERIMENTS["use_weight_calibration"]

                        for use_weight_calibration in calibration_options:
                            # Apply covariate constraints
                            if estimator in CONSTRAINTS.get("never_covariates", set()):
                                covariate_options = [False]
                            else:
                                covariate_options = EXPERIMENTS["use_covariates"]

                            for use_covariates in covariate_options:
                                var_cap = EXPERIMENTS["var_cap"]

                                spec = ExperimentSpec(
                                    ablation="unified",
                                    dataset_path=str(DATA_PATH),
                                    estimator=estimator,
                                    sample_size=sample_size,
                                    oracle_coverage=oracle_coverage,
                                    rho=var_cap,
                                    n_seeds=1,
                                    seed_base=seed,
                                    extra={
                                        "use_weight_calibration": use_weight_calibration,
                                        "reward_calibration_mode": EXPERIMENTS[
                                            "reward_calibration_mode"
                                        ],
                                        "use_covariates": use_covariates,
                                        # CF-bits computation removed from library
                                        "var_cap": var_cap,
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
        """Run ablation experiments in parallel.

        Args:
            max_workers: Maximum number of parallel workers

        Returns:
            List of result dictionaries
        """
        # Generate all specs
        all_specs = self._generate_all_specs()

        if not all_specs:
            logger.info("All experiments already completed!")
            return []

        # Default to 8 workers if not specified
        if max_workers is None:
            max_workers = 8

        all_results = []
        completed = 0
        failed = 0

        logger.info("=" * 60)
        logger.info(f"Starting parallel execution with {max_workers} workers")
        logger.info(f"Total experiments to run: {len(all_specs)}")
        logger.info(f"Already completed (cached): {len(self.completed_experiments)}")
        logger.info("=" * 60)

        # Create progress bar
        with tqdm(total=len(all_specs), desc="Experiments") as pbar:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_spec = {
                    executor.submit(
                        run_single_experiment_worker, spec.to_dict(), seed
                    ): (spec, seed)
                    for spec, seed in all_specs
                }

                # Process results as they complete
                for future in as_completed(future_to_spec):
                    spec, seed = future_to_spec[future]

                    try:
                        result = future.result()

                        # Save result
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

                        # Save failed result
                        failed_result = {
                            "spec": spec.to_dict(),
                            "success": False,
                            "error": str(e),
                        }
                        self._save_checkpoint(failed_result)
                        all_results.append(failed_result)

                    pbar.update(1)

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("UNIFIED ABLATION COMPLETE")
        logger.info(f"Completed: {completed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Results saved to: {self.results_file}")
        logger.info("=" * 60)

        return all_results

    def _run_ablation_sequential(self) -> List[Dict[str, Any]]:
        """Run ablation experiments sequentially (original implementation)."""
        all_results = []
        total_experiments = 0
        skipped = 0
        completed = 0
        failed = 0

        # SEEDS AS OUTERMOST LOOP - ensures all estimators are evaluated with seed 0 first
        for seed in EXPERIMENTS["seeds"]:
            logger.info(f"\n{'='*60}")
            logger.info(f"STARTING SEED {seed}")
            logger.info(f"{'='*60}")

            # Generate all parameter combinations for this seed
            for estimator in EXPERIMENTS["estimators"]:
                for sample_size in EXPERIMENTS["sample_sizes"]:
                    for oracle_coverage in EXPERIMENTS["oracle_coverages"]:

                        # Apply calibration constraints
                        if estimator in CONSTRAINTS.get("requires_calibration", set()):
                            # These estimators MUST have calibration
                            calibration_options = [True]
                        elif estimator in CONSTRAINTS.get("never_calibrated", set()):
                            # These estimators NEVER use calibration
                            calibration_options = [False]
                        else:
                            # These estimators can use either
                            calibration_options = EXPERIMENTS["use_weight_calibration"]

                        for use_weight_calibration in calibration_options:
                            # Apply covariate constraints
                            if estimator in CONSTRAINTS.get("never_covariates", set()):
                                covariate_options = [False]
                            else:
                                covariate_options = EXPERIMENTS["use_covariates"]

                            for use_covariates in covariate_options:
                                # Get fixed var_cap value from config
                                var_cap = EXPERIMENTS["var_cap"]

                                # Create specification
                                spec = ExperimentSpec(
                                    ablation="unified",
                                    dataset_path=str(DATA_PATH),
                                    estimator=estimator,
                                    sample_size=sample_size,
                                    oracle_coverage=oracle_coverage,
                                    rho=var_cap,  # Set rho field directly (var_cap is the variance budget)
                                    n_seeds=1,  # Single seed per experiment
                                    seed_base=seed,  # Use current seed from iteration
                                    extra={
                                        "use_weight_calibration": use_weight_calibration,
                                        "reward_calibration_mode": EXPERIMENTS[
                                            "reward_calibration_mode"
                                        ],
                                        "use_covariates": use_covariates,
                                        # CF-bits computation removed from library
                                        "var_cap": var_cap,  # Also keep in extra for backward compatibility
                                    },
                                )

                                # Check if already completed
                                exp_id = self._generate_exp_id(spec.to_dict())
                                if exp_id in self.completed_experiments:
                                    skipped += 1
                                    logger.debug(f"Skipping completed: {exp_id}")
                                    continue

                                total_experiments += 1

                                # Run experiment with current seed
                                logger.info(
                                    f"\n[Seed {seed}][{completed + failed + 1}/{total_experiments}] "
                                    f"Running: {estimator} n={sample_size} "
                                    f"oracle={oracle_coverage:.0%} "
                                    f"weight_cal={use_weight_calibration} "
                                    f"covariates={use_covariates}"
                                )

                                try:
                                    # Run with current seed
                                    result = self.run_single(spec, seed)

                                    # Save result
                                    self._save_checkpoint(result)
                                    all_results.append(result)
                                    completed += 1

                                    # Log summary
                                    if result.get("success"):
                                        if "rmse_vs_oracle" in result:
                                            logger.info(
                                                f"  ✓ RMSE: {result['rmse_vs_oracle']:.4f}"
                                            )
                                        else:
                                            logger.info("  ✓ Completed")

                                except Exception as e:
                                    logger.error(f"  ✗ Failed: {e}")
                                    failed += 1

                                    # Save failed result
                                    failed_result = {
                                        "spec": spec.to_dict(),
                                        "success": False,
                                        "error": str(e),
                                    }
                                    self._save_checkpoint(failed_result)

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("UNIFIED ABLATION COMPLETE")
        logger.info(f"Completed: {completed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Skipped (cached): {skipped}")
        logger.info(f"Results saved to: {self.results_file}")
        logger.info("=" * 60)

        return all_results


def main() -> None:
    """Run unified ablation experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run unified ablation experiments")
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
    args = parser.parse_args()

    ablation = UnifiedAblation()
    results = ablation.run_ablation(
        parallel=not args.sequential, max_workers=args.workers
    )

    # Save final aggregated results
    if results:
        summary_file = Path(RESULTS_PATH).parent / "unified_summary.json"
        with open(summary_file, "w") as f:
            # Convert numpy types before saving
            summary_data = UnifiedAblation._convert_numpy(
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
