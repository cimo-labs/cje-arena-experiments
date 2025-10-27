"""Standardized schemas for ablation experiments."""

import json
import time
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
import subprocess


@dataclass(frozen=True)
class ExperimentSpec:
    """Standardized input specification for ablation experiments.

    This ensures every ablation speaks the same language and enables
    caching, reproducibility, and easy aggregation.
    """

    # Required fields
    ablation: str  # e.g., "oracle_coverage"
    dataset_path: str  # Path to CJE dataset
    estimator: str  # e.g., "calibrated-ips", "dr-cpo"

    # Data configuration
    oracle_coverage: Optional[float] = None  # Fraction of oracle labels (0-1)
    sample_size: Optional[int] = None  # Number of samples to use
    sample_fraction: Optional[float] = None  # Alternative: fraction of dataset

    # Method parameters
    rho: Optional[float] = None  # Variance cap for SIMCal
    lambda_tr: Optional[float] = None  # Trust region tempering
    ordering: Optional[str] = None  # "S", "R", "S_bucket_shuffle"
    draws_per_prompt: Optional[int] = None  # Fresh draws for DR (K)

    # Temporal analysis
    temporal_block: Optional[int] = None  # Block size for bootstrap

    # Experiment configuration
    n_seeds: int = 5  # Number of random seeds
    seed_base: int = 42  # Base seed for reproducibility

    # Additional parameters
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()[:8]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def create_result(spec: ExperimentSpec, seed: int) -> Dict[str, Any]:
    """Create standardized result dictionary.

    Args:
        spec: Experiment specification
        seed: Random seed for this run

    Returns:
        Dictionary with all fields initialized
    """
    return {
        # Identity and provenance
        "seed": seed,
        "spec": spec.to_dict(),
        "git_commit": get_git_commit(),
        # Execution metadata
        "start_ts": time.time(),
        "runtime_s": None,
        "success": False,
        "error": None,
        # Core estimation results (per policy)
        "estimates": {},  # Policy -> estimate
        "standard_errors": {},  # Policy -> SE
        "confidence_intervals": {},  # Policy -> (lower, upper)
        "oracle_truths": {},  # Policy -> oracle ground truth
        # Weight diagnostics (per policy)
        "ess_absolute": {},  # Effective sample size (absolute)
        "ess_relative": {},  # ESS / n (percentage)
        "max_weight": {},  # Maximum single weight
        "tail_alpha": {},  # Hill estimator of tail index
        "weight_cv": {},  # Coefficient of variation
        "hellinger_affinity": {},  # Structural overlap metric (cannot be improved by calibration)
        "mass_concentration": {},  # Fraction of weights near zero
        # Calibration metrics
        "calibration_rmse": None,  # RMSE of judge→oracle calibration
        "calibration_r2": None,  # R² of judge→oracle calibration (more interpretable)
        "calibrated_reward_min": None,  # Min calibrated reward on oracle slice (detects overlap issues!)
        "calibrated_reward_max": None,  # Max calibrated reward on oracle slice
        "reward_overlap_warning": None,  # True if min > 0.1 or max < 0.9 (indicates extrapolation issues)
        "reward_calibration_used": None,  # Which mode was used (auto may select)
        # "simcal_distortion": {},  # Not currently computed
        # Oracle augmentation metrics
        "oracle_slice_size": None,  # Number of oracle labels used
        # DR-specific metrics
        "mc_variance_share": {},  # Monte Carlo share of variance
        "outcome_r2_min": None,  # Minimum outcome model R² across policies
        "outcome_r2_max": None,  # Maximum outcome model R² across policies
        "draws_per_prompt": None,  # K value used
        "policies_skipped": [],  # Policies that couldn't be estimated
        # Overall metrics
        "overall_status": None,  # Health check: "GOOD", "WARNING", or "CRITICAL"
        "rmse_vs_oracle": None,  # RMSE across all policies
        "mean_ci_width": None,  # Average CI width
        "n_samples": None,  # Actual number of samples used
        "n_oracle": None,  # Actual number of oracle labels
        # CF-bits metrics removed - feature no longer in library
    }


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results across seeds.

    Args:
        results: List of result dictionaries from same spec

    Returns:
        Aggregated statistics
    """
    import numpy as np

    if not results:
        return {}

    # Extract successful results
    successful = [r for r in results if r.get("success", False)]
    if not successful:
        return {
            "n_seeds_total": len(results),
            "n_seeds_successful": 0,
            "all_failed": True,
            "errors": [r.get("error", "Unknown") for r in results],
        }

    # Aggregate estimates across seeds
    aggregated = {
        "spec": results[0]["spec"],
        "n_seeds_total": len(results),
        "n_seeds_successful": len(successful),
        "estimates_mean": {},
        "estimates_se": {},
        "estimates_ci": {},
        "oracle_truths": successful[0].get("oracle_truths", {}),
    }

    # Get all policies
    policies = set()
    for r in successful:
        if "estimates" in r:
            policies.update(r["estimates"].keys())

    # Aggregate each policy
    for policy in policies:
        estimates = []
        ses = []

        for r in successful:
            if policy in r.get("estimates", {}):
                est = r["estimates"][policy]
                if est == est:  # Not NaN
                    estimates.append(est)

                    if "standard_errors" in r and policy in r["standard_errors"]:
                        se = r["standard_errors"][policy]
                        if se == se:  # Not NaN
                            ses.append(se)

        if estimates:
            mean_est = np.mean(estimates)

            # Combined SE (within + between seed variance)
            if ses and len(ses) == len(estimates):
                within_var = np.mean([s**2 for s in ses])
                between_var = np.var(estimates, ddof=1) if len(estimates) > 1 else 0
                combined_se = np.sqrt(within_var + between_var)
            else:
                combined_se = (
                    np.std(estimates, ddof=1) / np.sqrt(len(estimates))
                    if len(estimates) > 1
                    else 0
                )

            aggregated["estimates_mean"][policy] = mean_est
            aggregated["estimates_se"][policy] = combined_se
            aggregated["estimates_ci"][policy] = (
                mean_est - 1.96 * combined_se,
                mean_est + 1.96 * combined_se,
            )

    # Add diagnostic aggregates
    aggregated["mean_ess"] = np.mean(
        [
            np.mean(list(r.get("ess_absolute", {}).values()))
            for r in successful
            if r.get("ess_absolute")
        ]
    )

    aggregated["policies_skipped"] = list(
        set().union(*[set(r.get("policies_skipped", [])) for r in successful])
    )

    return aggregated
