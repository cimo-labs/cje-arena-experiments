"""
Experiment configuration for Preferences ablation study.

Compares CJE (continuous calibration) vs Rogan-Gladen (binary adjustment)
on pairwise human preference data from Chatbot Arena.

This is the "bulletproof" experiment addressing the reviewer escape hatch:
"Lee et al. is for preferences, MBPP is code."
"""

import numpy as np
from pathlib import Path

# Core experiment parameters for Preferences
EXPERIMENTS_PREFERENCES = {
    "estimators": [
        "naive-direct",  # Raw judge scores, no calibration (baseline)
        "direct",        # Monotone calibration (CJE)
        # Binary methods are handled separately in the ablation runner
    ],
    # Sample sizes (Arena has ~10k pairs after filtering)
    "sample_sizes": [2000, 5000, 10000],
    # Oracle coverage fractions to test
    "oracle_coverages": [0.01, 0.02, 0.05, 0.10, 0.25],
    # Weight calibration not used for Direct mode
    "use_weight_calibration": [False],
    # Covariates off by default for preferences (no obvious response covariate)
    "use_covariates": [False],
    # Reward calibration mode
    "reward_calibration_mode": "monotone",
    # Multiple seeds for robust results
    "seeds": np.arange(0, 30, 1),
    # Variance budget (not used for Direct, but needed for schema)
    "var_cap": 1.0,
    # Inference method: "bootstrap" for proper UQ with transport correction
    "inference_method": "bootstrap",
    # Bootstrap samples
    "n_bootstrap": 500,
    # Calibration policy: fit calibrator only on base, residual corrections on all
    # This enables transport-aware bias correction for target policies
    "calibration_policy": "base",
}

# Method-specific constraints for Preferences (Direct mode only)
NEVER_CALIBRATED_PREFERENCES = {
    "direct",        # Uses reward calibration, but no weight calibration
    "naive-direct",  # By definition: no calibration at all
}

NEVER_COVARIATES_PREFERENCES = {
    "naive-direct",  # Uses raw judge scores, covariates have no effect
}

CONSTRAINTS_PREFERENCES = {
    "never_calibrated": NEVER_CALIBRATED_PREFERENCES,
    "never_covariates": NEVER_COVARIATES_PREFERENCES,
}

# Target policies for Preferences (biased judge variants - excludes base which is logging policy)
TARGET_POLICIES_PREFERENCES = ["verbose", "terse", "contrarian"]

# All policies including base (for binary baseline which processes all)
ALL_POLICIES_PREFERENCES = ["base", "verbose", "terse", "contrarian"]

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data_preferences"
DATA_PATH_PREFERENCES = DATA_DIR / "preferences_dataset.jsonl"
RESPONSES_DIR_PREFERENCES = DATA_DIR / "responses"
RESULTS_DIR_PREFERENCES = BASE_DIR / "results_preferences"
RESULTS_PATH_PREFERENCES = RESULTS_DIR_PREFERENCES / "all_experiments.jsonl"
CHECKPOINT_PATH_PREFERENCES = RESULTS_DIR_PREFERENCES / "checkpoint.jsonl"

# Binary baseline output
BINARY_RESULTS_PATH = RESULTS_DIR_PREFERENCES / "binary_baseline.jsonl"
BINARY_SUMMARY_PATH = RESULTS_DIR_PREFERENCES / "binary_baseline_summary.json"

# DR config (for cross-fitting in calibration)
DR_CONFIG_PREFERENCES = {
    "n_folds": 5,
}

# Runtime configuration
RUNTIME_PREFERENCES = {
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
    exp = EXPERIMENTS_PREFERENCES

    n_estimators = len(exp["estimators"])
    n_sizes = len(exp["sample_sizes"])
    n_coverages = len(exp["oracle_coverages"])
    n_cov_settings = len(exp["use_covariates"])
    n_seeds = len(exp["seeds"])

    # Base count
    base_count = n_estimators * n_sizes * n_coverages * n_cov_settings * n_seeds

    return base_count


def get_binary_experiment_count() -> int:
    """Calculate total number of binary baseline experiments.

    Each experiment processes all policies with both naive and adjusted methods.
    """
    exp = EXPERIMENTS_PREFERENCES
    n_sizes = len(exp["sample_sizes"])
    n_coverages = len(exp["oracle_coverages"])
    n_seeds = len(exp["seeds"])

    # One experiment per (sample_size, oracle_coverage, seed)
    # Each experiment computes naive and adjusted for all policies
    return n_sizes * n_coverages * n_seeds


if __name__ == "__main__":
    print("Preferences Ablation Configuration")
    print("=" * 40)
    print(f"Estimators: {EXPERIMENTS_PREFERENCES['estimators']}")
    print(f"Sample sizes: {EXPERIMENTS_PREFERENCES['sample_sizes']}")
    print(f"Oracle coverages: {EXPERIMENTS_PREFERENCES['oracle_coverages']}")
    print(f"Covariate settings: {EXPERIMENTS_PREFERENCES['use_covariates']}")
    print(f"Seeds: {len(EXPERIMENTS_PREFERENCES['seeds'])}")
    print(f"\nTotal continuous experiments: {get_experiment_count():,}")
    print(f"Total binary experiments: {get_binary_experiment_count():,}")
    print(f"\nData path: {DATA_PATH_PREFERENCES}")
    print(f"Results path: {RESULTS_PATH_PREFERENCES}")
