"""
Experiment configuration for unified ablation system.

This defines all parameter combinations we want to test.
"""

import numpy as np

# Core experiment parameters
EXPERIMENTS = {
    "estimators": [
        "direct",  # Direct method (on-policy evaluation using fresh draws)
        "naive-direct",  # Direct method WITHOUT calibration (raw judge scores)
        "raw-ips",  # SNIPS (Self-Normalized IPS, no weight calibration)
        "calibrated-ips",  # Calibrated IPS
        # "orthogonalized-ips",  # Orthogonalized Calibrated IPS
        "dr-cpo",  # DR-CPO
        # "oc-dr-cpo",  # Orthogonalized Calibrated DR
        "tr-cpo-e",  # Triply-Robust CPO (efficient, m̂(S))
        # "tr-cpo-e-anchored-orthogonal",  # TR-CPO (efficient + anchored + orthogonal)
        "stacked-dr",  # Ensemble with dr-cpo, tmle, mrdr
    ],
    "sample_sizes": [250, 500, 1000, 2500, 5000],
    "oracle_coverages": [0.05, 0.10, 0.25, 0.5, 1.00],
    # Key ablation: calibration on/off
    "use_weight_calibration": [
        True,
        False,
    ],  # Test with and without weight calibration (SIMCal)
    # Key ablation: covariates for reward calibration and outcome modeling
    "use_covariates": [
        False,
        True,
    ],  # Test with and without response_length covariate
    # Reward calibration mode (not ablated - just use monotone)
    "reward_calibration_mode": "auto",
    # Multiple seeds for robust results
    "seeds": np.arange(0, 50, 1),
    # CF-bits computation (disabled - feature removed from library)
    # Set to False to disable CF-bits metrics for all experiments
    "compute_cfbits": False,  # Disabled - CF-bits removed from library
    # Variance budget (rho) for SIMCal - fixed at 1.0 (doesn't bind in practice)
    # Controls maximum allowed variance: Var(W_calibrated) ≤ var_cap * Var(W_baseline)
    "var_cap": 1.0,  # Fixed at no variance increase (empirically doesn't bind)
}

# Method-specific constraints
from typing import Dict, Any

# These estimators REQUIRE calibration (can't be turned off)
REQUIRES_CALIBRATION = {
    "calibrated-ips",  # By definition
    "orthogonalized-ips",  # Requires calibrated weights for orthogonalization
    "oc-dr-cpo",  # Orthogonalized Calibrated DR requires calibration
    "stacked-dr",  # Production default - always uses calibration
}

# These estimators can work with or without calibration
CALIBRATION_OPTIONAL = {
    "dr-cpo",  # Can use raw or calibrated weights
}

# These estimators never use weight calibration
NEVER_CALIBRATED = {
    "direct",  # Direct method doesn't use importance weights
    "naive-direct",  # Also doesn't use importance weights
    "raw-ips",  # SNIPS never uses weight calibration by design
    "tr-cpo-e",  # Also uses raw/Hajek weights, but with m̂(S) in TR term for efficiency
    "tr-cpo-e-anchored-orthogonal",  # Uses raw weights with SIMCal anchoring + orthogonalization
}

# These estimators never use covariates (no reward calibration)
NEVER_COVARIATES = {
    "naive-direct",  # Uses raw judge scores, no calibration, so covariates have no effect
}

CONSTRAINTS = {
    "requires_calibration": REQUIRES_CALIBRATION,
    "calibration_optional": CALIBRATION_OPTIONAL,
    "never_calibrated": NEVER_CALIBRATED,
    "never_covariates": NEVER_COVARIATES,
}

# Fixed parameters for DR methods
DR_CONFIG = {
    "n_folds": 5,  # Standard k-fold cross-fitting (faster, still reliable)
    "v_folds_stacking": 5,  # Outer folds for stacked-dr
}

# CF-bits configuration removed - feature no longer in library

# Paths (absolute to avoid confusion)
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / "data" / "cje_dataset.jsonl"
RESULTS_PATH = BASE_DIR / "results" / "all_experiments.jsonl"
CHECKPOINT_PATH = BASE_DIR / "results" / "checkpoint.jsonl"

# Runtime configuration
RUNTIME = {
    "checkpoint_every": 10,  # Save progress every N experiments
    "verbose": True,
    "parallel": False,
    "max_workers": 10,
}
