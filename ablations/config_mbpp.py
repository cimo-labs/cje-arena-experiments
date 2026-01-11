"""
Experiment configuration for MBPP ablation study.

Direct mode only - no importance sampling estimators.
Tests calibration effect (naive-direct vs direct vs direct+cov) on coding tasks.
"""

import numpy as np
from pathlib import Path

# Core experiment parameters for MBPP
# Direct estimators only - no logprobs needed
EXPERIMENTS_MBPP = {
    "estimators": [
        "naive-direct",  # Raw judge scores, no calibration (baseline)
        "direct",        # Monotone calibration only
        # Note: direct+cov is achieved by setting use_covariates=True
    ],
    # Sample sizes (MBPP has ~974 problems)
    "sample_sizes": [200, 400, 600, 800, 974],
    # Oracle coverage fractions to test
    "oracle_coverages": [0.05, 0.10, 0.25, 0.50, 1.00],
    # Weight calibration not used for Direct mode
    "use_weight_calibration": [False],
    # Key ablation: covariates for reward calibration
    # When True, uses response_length as calibration covariate (direct+cov)
    "use_covariates": [False, True],
    # Reward calibration mode
    "reward_calibration_mode": "auto",
    # Multiple seeds for robust results
    "seeds": np.arange(0, 50, 1),
    # Variance budget (not used for Direct, but needed for schema)
    "var_cap": 1.0,
    # Inference method: "cluster_robust" for fast point estimates
    # Use "bootstrap" only for UQ experiments (much slower)
    "inference_method": "cluster_robust",
    # Bootstrap samples (only used if inference_method="bootstrap")
    "n_bootstrap": 500,
}

# Method-specific constraints for MBPP (Direct mode only)
NEVER_CALIBRATED_MBPP = {
    "direct",        # Uses reward calibration, but no weight calibration
    "naive-direct",  # By definition: no calibration at all
}

NEVER_COVARIATES_MBPP = {
    "naive-direct",  # Uses raw judge scores, covariates have no effect
}

CONSTRAINTS_MBPP = {
    "never_calibrated": NEVER_CALIBRATED_MBPP,
    "never_covariates": NEVER_COVARIATES_MBPP,
}

# Target policies for MBPP (all except base)
TARGET_POLICIES_MBPP = ["enhanced", "mini", "terse", "unhelpful"]

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data_mbpp"
DATA_PATH_MBPP = DATA_DIR / "mbpp_dataset.jsonl"
RESPONSES_DIR_MBPP = DATA_DIR / "responses"
RESULTS_DIR_MBPP = BASE_DIR / "results_mbpp"
RESULTS_PATH_MBPP = RESULTS_DIR_MBPP / "all_experiments.jsonl"
CHECKPOINT_PATH_MBPP = RESULTS_DIR_MBPP / "checkpoint.jsonl"

# DR config (for cross-fitting in calibration)
DR_CONFIG_MBPP = {
    "n_folds": 5,
}

# Runtime configuration
RUNTIME_MBPP = {
    "checkpoint_every": 10,
    "verbose": True,
    "parallel": True,
    "max_workers": 8,
}


def get_experiment_count() -> int:
    """Calculate total number of experiments.

    Returns:
        Total experiment count
    """
    exp = EXPERIMENTS_MBPP

    n_estimators = len(exp["estimators"])
    n_sizes = len(exp["sample_sizes"])
    n_coverages = len(exp["oracle_coverages"])
    n_cov_settings = len(exp["use_covariates"])
    n_seeds = len(exp["seeds"])

    # Base count
    base_count = n_estimators * n_sizes * n_coverages * n_cov_settings * n_seeds

    # Subtract invalid combinations (naive-direct with use_covariates=True)
    # naive-direct doesn't use calibration, so covariates have no effect
    invalid_naive = n_sizes * n_coverages * n_seeds  # One covariate setting skipped

    return base_count - invalid_naive


if __name__ == "__main__":
    print("MBPP Ablation Configuration")
    print("=" * 40)
    print(f"Estimators: {EXPERIMENTS_MBPP['estimators']}")
    print(f"Sample sizes: {EXPERIMENTS_MBPP['sample_sizes']}")
    print(f"Oracle coverages: {EXPERIMENTS_MBPP['oracle_coverages']}")
    print(f"Covariate settings: {EXPERIMENTS_MBPP['use_covariates']}")
    print(f"Seeds: {len(EXPERIMENTS_MBPP['seeds'])}")
    print(f"\nTotal experiments: {get_experiment_count():,}")
    print(f"\nData path: {DATA_PATH_MBPP}")
    print(f"Results path: {RESULTS_PATH_MBPP}")
