#!/usr/bin/env python3
"""
Distribution shift experiment: CJE vs Rogan-Gladen under Pr(Z) shift.

This tests Lee et al.'s claim (Section 7) that their estimator is robust to
distribution shift in Pr(Z) between calibration and test sets.

Theoretical prediction:
- Rogan-Gladen estimates Pr(Ẑ|Z) which doesn't depend on Pr(Z) → should be unbiased
- CJE learns E[Y|S] which DOES depend on Pr(Z) via Bayes' rule → may be biased

Usage:
    python run_distribution_shift.py
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

# Add parent directories to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from binary_baseline import (
    estimate_confusion_stats,
    rogan_gladen_estimate,
    lee_et_al_confidence_interval,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
SHIFT_RATIOS = [0.30, 0.40, 0.50, 0.60, 0.70]  # Pr(Z=1) in calibration
TEST_PZ1 = None  # Use natural rate in test (will be ~0.50)
SEEDS = list(range(30))
CALIB_SIZE = 500  # Fixed calibration budget
TEST_SIZE = 5000  # Large test set
THRESHOLD = 0.5  # Binary threshold for Rogan-Gladen

# Paths
RESPONSES_DIR = Path(__file__).parent.parent / "data_preferences" / "responses"
RESULTS_DIR = Path(__file__).parent / "results_preferences"


@dataclass
class ShiftResult:
    """Result of a single distribution shift experiment."""
    shift_ratio: float
    seed: int

    # Ground truth
    test_pz1: float  # Actual Pr(Z=1) in test
    calib_pz1: float  # Actual Pr(Z=1) in calibration (after shifting)
    oracle_mean: float  # True E[Y] in test

    # CJE results
    cje_estimate: float
    cje_bias: float

    # Rogan-Gladen results
    rg_estimate: float
    rg_bias: float
    rg_valid: bool
    rg_se: float
    rg_sp: float
    rg_j: float


def load_preference_data(policy: str = "base") -> List[Dict]:
    """Load preference response data for a policy."""
    response_file = RESPONSES_DIR / f"{policy}_responses.jsonl"
    if not response_file.exists():
        raise FileNotFoundError(f"Missing response file: {response_file}")

    records = []
    with open(response_file, 'r') as f:
        for line in f:
            records.append(json.loads(line))

    return records


def extract_data(records: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract oracle labels, judge scores, and binary predictions from records.

    Returns:
        y_oracle: Oracle labels (0/1)
        s_score: Continuous judge scores (0-1)
        z_binary: Binary predictions (0/1 from thresholding)
    """
    y_oracle = []
    s_score = []

    for r in records:
        meta = r.get('metadata', {})
        oracle = meta.get('oracle_label')
        score = meta.get('judge_score')

        if oracle is not None and score is not None:
            y_oracle.append(oracle)
            s_score.append(score)

    y_oracle = np.array(y_oracle)
    s_score = np.array(s_score)
    z_binary = (s_score >= THRESHOLD).astype(int)

    return y_oracle, s_score, z_binary


def create_shifted_calibration(
    y_oracle: np.ndarray,
    s_score: np.ndarray,
    z_binary: np.ndarray,
    target_pz1: float,
    n_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a calibration set with target Pr(Z=1).

    Subsamples from the data to achieve the desired class balance.
    """
    rng = np.random.RandomState(seed)

    pos_idx = np.where(y_oracle == 1)[0]
    neg_idx = np.where(y_oracle == 0)[0]

    # Calculate how many of each class
    n_pos = int(n_samples * target_pz1)
    n_neg = n_samples - n_pos

    # Sample with replacement if needed
    if n_pos > len(pos_idx):
        pos_sample = rng.choice(pos_idx, size=n_pos, replace=True)
    else:
        pos_sample = rng.choice(pos_idx, size=n_pos, replace=False)

    if n_neg > len(neg_idx):
        neg_sample = rng.choice(neg_idx, size=n_neg, replace=True)
    else:
        neg_sample = rng.choice(neg_idx, size=n_neg, replace=False)

    # Combine and shuffle
    sample_idx = np.concatenate([pos_sample, neg_sample])
    rng.shuffle(sample_idx)

    return y_oracle[sample_idx], s_score[sample_idx], z_binary[sample_idx]


def fit_cje_and_estimate(
    calib_y: np.ndarray,
    calib_s: np.ndarray,
    test_s: np.ndarray,
) -> float:
    """Fit CJE isotonic calibration and estimate on test set.

    CJE learns f(S) ≈ E[Y|S] via isotonic regression, then estimates E[f(S)] on test.
    """
    # Fit isotonic regression: S -> Y
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(calib_s, calib_y)

    # Predict calibrated values on test set
    calibrated = iso.predict(test_s)

    # Estimate is mean of calibrated values
    return calibrated.mean()


def fit_rg_and_estimate(
    calib_y: np.ndarray,
    calib_z: np.ndarray,
    test_z: np.ndarray,
) -> Tuple[float, bool, float, float, float]:
    """Fit Rogan-Gladen confusion matrix and estimate on test set.

    RG estimates Se=P(Ẑ=1|Y=1) and Sp=P(Ẑ=0|Y=0), then adjusts:
        θ̂ = (p̂ + Sp - 1) / (Se + Sp - 1)

    Returns:
        estimate, valid, se, sp, j
    """
    # Estimate confusion matrix from calibration
    stats = estimate_confusion_stats(calib_y, calib_z)

    # Compute test proportion
    p_hat = test_z.mean()

    # Adjust using Rogan-Gladen
    estimate, valid = rogan_gladen_estimate(
        p_hat, stats.sensitivity, stats.specificity
    )

    return estimate, valid, stats.sensitivity, stats.specificity, stats.youdens_j


def run_single_experiment(
    y_oracle: np.ndarray,
    s_score: np.ndarray,
    z_binary: np.ndarray,
    shift_ratio: float,
    seed: int,
) -> ShiftResult:
    """Run a single distribution shift experiment."""
    rng = np.random.RandomState(seed)

    n_total = len(y_oracle)

    # Split into potential calibration pool and test set
    # Use stratified split to maintain natural distribution in test
    idx = np.arange(n_total)
    rng.shuffle(idx)

    test_idx = idx[:TEST_SIZE]
    calib_pool_idx = idx[TEST_SIZE:]

    # Test set with natural distribution
    test_y = y_oracle[test_idx]
    test_s = s_score[test_idx]
    test_z = z_binary[test_idx]
    test_pz1 = test_y.mean()
    oracle_mean = test_y.mean()

    # Create shifted calibration set
    pool_y = y_oracle[calib_pool_idx]
    pool_s = s_score[calib_pool_idx]
    pool_z = z_binary[calib_pool_idx]

    calib_y, calib_s, calib_z = create_shifted_calibration(
        pool_y, pool_s, pool_z,
        target_pz1=shift_ratio,
        n_samples=CALIB_SIZE,
        seed=seed + 1000,  # Different seed for calibration sampling
    )
    calib_pz1 = calib_y.mean()

    # Fit CJE and estimate
    cje_estimate = fit_cje_and_estimate(calib_y, calib_s, test_s)
    cje_bias = cje_estimate - oracle_mean

    # Fit Rogan-Gladen and estimate
    rg_estimate, rg_valid, rg_se, rg_sp, rg_j = fit_rg_and_estimate(
        calib_y, calib_z, test_z
    )
    rg_bias = rg_estimate - oracle_mean if rg_valid else np.nan

    return ShiftResult(
        shift_ratio=shift_ratio,
        seed=seed,
        test_pz1=test_pz1,
        calib_pz1=calib_pz1,
        oracle_mean=oracle_mean,
        cje_estimate=cje_estimate,
        cje_bias=cje_bias,
        rg_estimate=rg_estimate,
        rg_bias=rg_bias,
        rg_valid=rg_valid,
        rg_se=rg_se,
        rg_sp=rg_sp,
        rg_j=rg_j,
    )


def run_distribution_shift_experiment() -> Dict[str, Any]:
    """Run full distribution shift experiment."""
    logger.info("Loading preference data...")
    records = load_preference_data("base")
    logger.info(f"Loaded {len(records)} records")

    y_oracle, s_score, z_binary = extract_data(records)
    logger.info(f"Extracted {len(y_oracle)} samples with complete data")
    logger.info(f"Natural Pr(Z=1) = {y_oracle.mean():.3f}")

    results = []

    for shift_ratio in SHIFT_RATIOS:
        logger.info(f"\n=== Shift ratio: {shift_ratio:.2f} ===")

        for seed in tqdm(SEEDS, desc=f"Shift {shift_ratio}"):
            result = run_single_experiment(
                y_oracle, s_score, z_binary,
                shift_ratio=shift_ratio,
                seed=seed,
            )
            results.append(result)

    # Aggregate results
    summary = aggregate_results(results)

    # Save results
    output_file = RESULTS_DIR / "distribution_shift.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {output_file}")

    # Print summary
    print_summary(summary)

    return summary


def aggregate_results(results: List[ShiftResult]) -> Dict[str, Any]:
    """Aggregate results by shift ratio."""
    by_shift = {}

    for shift_ratio in SHIFT_RATIOS:
        shift_results = [r for r in results if r.shift_ratio == shift_ratio]

        cje_biases = [r.cje_bias for r in shift_results]
        rg_biases = [r.rg_bias for r in shift_results if r.rg_valid]

        by_shift[str(shift_ratio)] = {
            'n_experiments': len(shift_results),
            'n_valid_rg': len(rg_biases),
            'mean_calib_pz1': np.mean([r.calib_pz1 for r in shift_results]),
            'mean_test_pz1': np.mean([r.test_pz1 for r in shift_results]),

            # CJE stats
            'cje_mean_bias': float(np.mean(cje_biases)),
            'cje_std_bias': float(np.std(cje_biases)),
            'cje_rmse': float(np.sqrt(np.mean(np.array(cje_biases)**2))),

            # RG stats
            'rg_mean_bias': float(np.mean(rg_biases)) if rg_biases else None,
            'rg_std_bias': float(np.std(rg_biases)) if rg_biases else None,
            'rg_rmse': float(np.sqrt(np.mean(np.array(rg_biases)**2))) if rg_biases else None,

            # Confusion matrix stats
            'mean_youdens_j': float(np.mean([r.rg_j for r in shift_results])),
        }

    return {
        'config': {
            'shift_ratios': SHIFT_RATIOS,
            'seeds': len(SEEDS),
            'calib_size': CALIB_SIZE,
            'test_size': TEST_SIZE,
            'threshold': THRESHOLD,
        },
        'results': by_shift,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    """Print a summary table of results."""
    print("\n" + "="*80)
    print("DISTRIBUTION SHIFT EXPERIMENT RESULTS")
    print("="*80)
    print(f"\nConfig: {CALIB_SIZE} calibration, {TEST_SIZE} test, {len(SEEDS)} seeds")
    print(f"Shift ratios: {SHIFT_RATIOS}")
    print()

    print(f"{'Calib Pr(Z=1)':<15} | {'CJE Bias':<15} | {'RG Bias':<15} | {'CJE RMSE':<10} | {'RG RMSE':<10}")
    print("-"*80)

    for shift_ratio in SHIFT_RATIOS:
        r = summary['results'][str(shift_ratio)]
        cje_bias = f"{r['cje_mean_bias']:+.4f} ± {r['cje_std_bias']:.4f}"
        rg_bias = f"{r['rg_mean_bias']:+.4f} ± {r['rg_std_bias']:.4f}" if r['rg_mean_bias'] is not None else "N/A"
        cje_rmse = f"{r['cje_rmse']:.4f}"
        rg_rmse = f"{r['rg_rmse']:.4f}" if r['rg_rmse'] is not None else "N/A"

        print(f"{shift_ratio:<15.2f} | {cje_bias:<15} | {rg_bias:<15} | {cje_rmse:<10} | {rg_rmse:<10}")

    print()
    print("Interpretation:")
    print("- If RG bias ≈ 0 across all shifts: RG is robust to distribution shift (as Lee et al. claim)")
    print("- If CJE bias varies with shift: CJE is affected by distribution shift")
    print()


if __name__ == "__main__":
    run_distribution_shift_experiment()
