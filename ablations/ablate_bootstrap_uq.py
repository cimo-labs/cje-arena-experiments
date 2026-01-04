#!/usr/bin/env python3
"""
Bootstrap UQ Ablation Study for Direct Mode.

Compares three uncertainty quantification methods:
1. Cluster-robust SE (baseline, no OUA)
2. OUA Jackknife (current default)
3. Bootstrap with calibrator refit (new)

Evaluates across oracle coverages, sample sizes, and bootstrap parameters.
"""

import json
import logging
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

# Import configuration
from config import DR_CONFIG, DATA_PATH

# Import existing ablation infrastructure
from core.base import BaseAblation
from core.schemas import ExperimentSpec, aggregate_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Ablation-specific configuration
BOOTSTRAP_UQ_CONFIG = {
    # Only test Direct mode
    "estimator": "direct",
    # Sample sizes to evaluate
    "sample_sizes": [250, 500, 1000, 2500],
    # Oracle coverages to evaluate
    "oracle_coverages": [0.05, 0.10, 0.25, 0.50],
    # Bootstrap replicate counts to test (for bootstrap method only)
    "n_bootstrap_values": [200, 500, 2000],
    # Number of random seeds for coverage estimation
    "seeds": list(range(50)),
    # UQ methods to compare
    "uq_methods": [
        {"inference_method": "cluster_robust", "oua_jackknife": False, "label": "cluster_robust"},
        {"inference_method": "cluster_robust", "oua_jackknife": True, "label": "oua_jackknife"},
        {"inference_method": "bootstrap", "oua_jackknife": False, "label": "bootstrap"},
    ],
}

# Paths
BASE_DIR = Path(__file__).parent
RESULTS_PATH = BASE_DIR / "results" / "bootstrap_uq.jsonl"
CHECKPOINT_PATH = BASE_DIR / "results" / "bootstrap_uq_checkpoint.jsonl"


def run_single_experiment_worker(
    spec_dict: Dict[str, Any], seed: int
) -> Dict[str, Any]:
    """Worker function for parallel execution of experiments."""
    from core.base import BaseAblation
    from core.schemas import ExperimentSpec

    ablation = BootstrapUQAblation()
    spec = ExperimentSpec(**spec_dict)

    try:
        result = ablation.run_single(spec, seed)
        return result
    except Exception as e:
        logger.error(f"Failed to run experiment: {e}")
        return {
            "spec": spec.to_dict(),
            "success": False,
            "error": str(e),
        }


class BootstrapUQAblation(BaseAblation):
    """Compare UQ methods: cluster-robust vs OUA jackknife vs bootstrap."""

    def __init__(self) -> None:
        super().__init__(name="bootstrap_uq")
        self.checkpoint_file = CHECKPOINT_PATH
        self.results_file = RESULTS_PATH
        self.completed_experiments = self._load_checkpoint()

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
        extra = spec.get("extra", {})
        key_params = [
            spec.get("estimator", ""),
            str(spec.get("sample_size", "")),
            str(spec.get("oracle_coverage", "")),
            str(extra.get("inference_method", "")),
            str(extra.get("oua_jackknife", "")),
            str(extra.get("n_bootstrap", "")),
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

    def generate_specs(self) -> List[Tuple[ExperimentSpec, int]]:
        """Generate all experiment specifications.

        Returns:
            List of (spec, seed) tuples
        """
        all_specs = []
        config = BOOTSTRAP_UQ_CONFIG

        for seed in config["seeds"]:
            for sample_size in config["sample_sizes"]:
                for oracle_coverage in config["oracle_coverages"]:
                    for method in config["uq_methods"]:
                        # For bootstrap, iterate over n_bootstrap values
                        if method["inference_method"] == "bootstrap":
                            for n_bootstrap in config["n_bootstrap_values"]:
                                spec = ExperimentSpec(
                                    ablation="bootstrap_uq",
                                    dataset_path=str(DATA_PATH),
                                    estimator=config["estimator"],
                                    sample_size=sample_size,
                                    oracle_coverage=oracle_coverage,
                                    n_seeds=1,
                                    seed_base=seed,
                                    extra={
                                        "inference_method": method["inference_method"],
                                        "oua_jackknife": method["oua_jackknife"],
                                        "n_bootstrap": n_bootstrap,
                                        "uq_label": f"bootstrap_n{n_bootstrap}",
                                    },
                                )

                                exp_id = self._generate_exp_id(spec.to_dict())
                                if exp_id not in self.completed_experiments:
                                    all_specs.append((spec, seed))
                        else:
                            # Non-bootstrap methods
                            spec = ExperimentSpec(
                                ablation="bootstrap_uq",
                                dataset_path=str(DATA_PATH),
                                estimator=config["estimator"],
                                sample_size=sample_size,
                                oracle_coverage=oracle_coverage,
                                n_seeds=1,
                                seed_base=seed,
                                extra={
                                    "inference_method": method["inference_method"],
                                    "oua_jackknife": method["oua_jackknife"],
                                    "n_bootstrap": None,
                                    "uq_label": method["label"],
                                },
                            )

                            exp_id = self._generate_exp_id(spec.to_dict())
                            if exp_id not in self.completed_experiments:
                                all_specs.append((spec, seed))

        return all_specs

    def run_ablation(
        self, parallel: bool = True, max_workers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Run the complete bootstrap UQ ablation.

        Args:
            parallel: Whether to run experiments in parallel
            max_workers: Max number of parallel workers (defaults to 8)

        Returns:
            List of result dictionaries
        """
        all_specs = self.generate_specs()

        if not all_specs:
            logger.info("All experiments already completed!")
            return []

        if max_workers is None:
            max_workers = 8

        all_results = []
        completed = 0
        failed = 0

        # Count expected experiments
        n_non_bootstrap = (
            len(BOOTSTRAP_UQ_CONFIG["sample_sizes"])
            * len(BOOTSTRAP_UQ_CONFIG["oracle_coverages"])
            * 2  # cluster_robust and oua_jackknife
            * len(BOOTSTRAP_UQ_CONFIG["seeds"])
        )
        n_bootstrap = (
            len(BOOTSTRAP_UQ_CONFIG["sample_sizes"])
            * len(BOOTSTRAP_UQ_CONFIG["oracle_coverages"])
            * len(BOOTSTRAP_UQ_CONFIG["n_bootstrap_values"])
            * len(BOOTSTRAP_UQ_CONFIG["seeds"])
        )
        total_expected = n_non_bootstrap + n_bootstrap

        logger.info("=" * 60)
        logger.info("BOOTSTRAP UQ ABLATION STUDY")
        logger.info("=" * 60)
        logger.info(f"Comparing 3 UQ methods for Direct mode:")
        logger.info("  1. cluster_robust - Cluster-robust SE (no OUA)")
        logger.info("  2. oua_jackknife  - OUA Jackknife (current default)")
        logger.info("  3. bootstrap      - Bootstrap with calibrator refit")
        logger.info("-" * 60)
        logger.info(f"Sample sizes: {BOOTSTRAP_UQ_CONFIG['sample_sizes']}")
        logger.info(f"Oracle coverages: {BOOTSTRAP_UQ_CONFIG['oracle_coverages']}")
        logger.info(f"Bootstrap counts: {BOOTSTRAP_UQ_CONFIG['n_bootstrap_values']}")
        logger.info(f"Seeds: {len(BOOTSTRAP_UQ_CONFIG['seeds'])}")
        logger.info("-" * 60)
        logger.info(f"Total experiments expected: {total_expected}")
        logger.info(f"Experiments remaining: {len(all_specs)}")
        logger.info(f"Already completed (cached): {len(self.completed_experiments)}")
        logger.info("=" * 60)

        if parallel:
            with tqdm(total=len(all_specs), desc="Experiments") as pbar:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_to_spec = {
                        executor.submit(
                            run_single_experiment_worker, spec.to_dict(), seed
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
                                uq_label = spec.extra.get("uq_label", "unknown")
                                pbar.set_postfix(
                                    done=completed,
                                    fail=failed,
                                    method=uq_label,
                                )
                            else:
                                failed += 1
                                logger.error(
                                    f"Failed: {result.get('error', 'Unknown error')}"
                                )

                        except Exception as e:
                            failed += 1
                            logger.error(f"Exception: {e}")

                            failed_result = {
                                "spec": spec.to_dict(),
                                "success": False,
                                "error": str(e),
                            }
                            self._save_checkpoint(failed_result)
                            all_results.append(failed_result)

                        pbar.update(1)
        else:
            # Sequential execution
            for spec, seed in tqdm(all_specs, desc="Experiments"):
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
                    failed_result = {
                        "spec": spec.to_dict(),
                        "success": False,
                        "error": str(e),
                    }
                    self._save_checkpoint(failed_result)
                    all_results.append(failed_result)

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("BOOTSTRAP UQ ABLATION COMPLETE")
        logger.info(f"Completed: {completed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Results saved to: {self.results_file}")
        logger.info("=" * 60)

        return all_results


def main() -> None:
    """Run bootstrap UQ ablation experiments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run bootstrap UQ ablation study for Direct mode"
    )
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
        "--output",
        type=str,
        default=None,
        help="Output file path (default: results/bootstrap_uq.jsonl)",
    )
    args = parser.parse_args()

    if args.output:
        global RESULTS_PATH, CHECKPOINT_PATH
        RESULTS_PATH = Path(args.output)
        CHECKPOINT_PATH = RESULTS_PATH.parent / "bootstrap_uq_checkpoint.jsonl"

    ablation = BootstrapUQAblation()
    results = ablation.run_ablation(
        parallel=not args.sequential, max_workers=args.workers
    )

    # Save summary
    if results:
        summary_file = RESULTS_PATH.parent / "bootstrap_uq_summary.json"
        with open(summary_file, "w") as f:
            summary_data = {
                "timestamp": time.time(),
                "n_experiments": len(results),
                "n_successful": sum(1 for r in results if r.get("success")),
                "config": {
                    "sample_sizes": BOOTSTRAP_UQ_CONFIG["sample_sizes"],
                    "oracle_coverages": BOOTSTRAP_UQ_CONFIG["oracle_coverages"],
                    "n_bootstrap_values": BOOTSTRAP_UQ_CONFIG["n_bootstrap_values"],
                    "n_seeds": len(BOOTSTRAP_UQ_CONFIG["seeds"]),
                },
            }
            json.dump(summary_data, f, indent=2)
        logger.info(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
