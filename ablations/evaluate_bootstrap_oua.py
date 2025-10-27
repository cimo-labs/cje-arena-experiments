"""Evaluate oracle bootstrap as alternative to jackknife for OUA.

This script compares:
1. Current K-fold jackknife (K=3-5)
2. Oracle bootstrap (B=50 resamples)

Goal: See if bootstrap gives better variance estimates and coverage.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from sklearn.isotonic import IsotonicRegression


@dataclass
class OUAComparison:
    """Results comparing jackknife vs bootstrap OUA."""
    oracle_size: int
    sample_size: int
    oracle_pct: float

    # Current jackknife
    jackknife_var: float
    jackknife_k: int

    # Bootstrap alternative
    bootstrap_var: float
    bootstrap_b: int

    # Ground truth (from multiple independent runs)
    empirical_var: float  # Variance across independent runs
    empirical_n_runs: int

    # Ratios
    jackknife_ratio: float  # jackknife_var / empirical_var
    bootstrap_ratio: float  # bootstrap_var / empirical_var


def oracle_bootstrap_variance(
    judge_scores: np.ndarray,
    oracle_labels: np.ndarray,
    oracle_mask: np.ndarray,
    all_scores: np.ndarray,
    n_bootstrap: int = 50,
    random_seed: int = 42
) -> float:
    """Compute oracle variance using bootstrap resampling.

    Args:
        judge_scores: Judge scores for oracle samples
        oracle_labels: Oracle labels
        oracle_mask: Boolean mask for oracle samples in full data
        all_scores: All judge scores to compute estimate on
        n_bootstrap: Number of bootstrap resamples (default 50)
        random_seed: Random seed

    Returns:
        Bootstrap variance estimate
    """
    rng = np.random.RandomState(random_seed)
    n_oracle = len(judge_scores)

    bootstrap_estimates = []

    for b in range(n_bootstrap):
        # Resample oracle samples with replacement
        boot_idx = rng.choice(n_oracle, size=n_oracle, replace=True)
        boot_scores = judge_scores[boot_idx]
        boot_labels = oracle_labels[boot_idx]

        # Fit isotonic regression on bootstrap sample
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(boot_scores, boot_labels)

        # Predict on all data and compute mean (the policy value estimate)
        calibrated = np.clip(iso.predict(all_scores), 0.0, 1.0)
        boot_estimate = float(np.mean(calibrated))
        bootstrap_estimates.append(boot_estimate)

    # Bootstrap variance
    boot_var = float(np.var(bootstrap_estimates, ddof=1))
    return boot_var


def oracle_jackknife_variance(
    judge_scores: np.ndarray,
    oracle_labels: np.ndarray,
    oracle_mask: np.ndarray,
    all_scores: np.ndarray,
    n_folds: int = 5,
    random_seed: int = 42
) -> Tuple[float, int]:
    """Compute oracle variance using K-fold jackknife (current method).

    Args:
        judge_scores: Judge scores for oracle samples
        oracle_labels: Oracle labels
        oracle_mask: Boolean mask for oracle samples in full data
        all_scores: All judge scores to compute estimate on
        n_folds: Number of folds
        random_seed: Random seed

    Returns:
        Tuple of (jackknife variance, K)
    """
    from sklearn.model_selection import KFold

    n_oracle = len(judge_scores)

    # Small oracle: use K=3
    if n_oracle < 50:
        n_folds = 3

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    jackknife_estimates = []

    for train_idx, test_idx in kf.split(judge_scores):
        # Train on all except this fold
        train_scores = judge_scores[train_idx]
        train_labels = oracle_labels[train_idx]

        # Fit isotonic on training fold
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(train_scores, train_labels)

        # Predict on all data
        calibrated = np.clip(iso.predict(all_scores), 0.0, 1.0)
        jk_estimate = float(np.mean(calibrated))
        jackknife_estimates.append(jk_estimate)

    K = len(jackknife_estimates)
    psi_bar = float(np.mean(jackknife_estimates))
    jk_var = (K - 1) / K * float(np.mean((np.array(jackknife_estimates) - psi_bar) ** 2))

    return jk_var, K


def load_experimental_data() -> List[Dict]:
    """Load experimental results to extract variance estimates.

    Returns:
        List of experimental records
    """
    results_path = Path("results/all_experiments.jsonl")

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    experiments = []
    with open(results_path) as f:
        for line in f:
            exp = json.loads(line)
            experiments.append(exp)

    return experiments


def group_by_config(experiments: List[Dict]) -> Dict[Tuple, List[Dict]]:
    """Group experiments by (estimator, oracle_size, sample_size, oracle_coverage).

    Returns:
        Dict mapping config tuple to list of experiments with that config
    """
    groups = {}

    for exp in experiments:
        spec = exp.get('spec', {})
        estimator = spec.get('estimator')
        sample_size = spec.get('sample_size')
        oracle_coverage = spec.get('oracle_coverage')

        # Get oracle size from metadata
        meta = exp.get('metadata', {})
        se_comp = meta.get('se_components', {})
        oracle_var_map = se_comp.get('oracle_variance_per_policy', {})

        # Extract oracle size if available
        oracle_size = None
        for policy, var in oracle_var_map.items():
            # We can infer oracle size from other metadata if needed
            # For now, compute from oracle_coverage * sample_size
            if oracle_coverage and sample_size:
                oracle_size = int(oracle_coverage * sample_size)
                break

        if estimator and sample_size and oracle_coverage is not None:
            key = (estimator, oracle_size, sample_size, oracle_coverage)
            if key not in groups:
                groups[key] = []
            groups[key].append(exp)

    return groups


def estimate_empirical_variance(experiments: List[Dict]) -> Tuple[float, int]:
    """Estimate empirical variance from multiple independent runs.

    Args:
        experiments: List of experiments with same config

    Returns:
        Tuple of (empirical variance, number of runs)
    """
    estimates = []

    for exp in experiments:
        # Get the point estimate for the first policy
        est = exp.get('estimates')
        if est and len(est) > 0:
            estimates.append(est[0])

    if len(estimates) < 2:
        return 0.0, len(estimates)

    emp_var = float(np.var(estimates, ddof=1))
    return emp_var, len(estimates)


def main():
    """Compare jackknife vs bootstrap OUA."""

    print("=" * 80)
    print("Evaluating Bootstrap OUA vs Jackknife OUA")
    print("=" * 80)
    print()

    # For this evaluation, we'll use simulated data that matches
    # the experimental setup, since we don't have raw scores in results

    # Simulate oracle scenarios matching our experiments
    scenarios = [
        # (oracle_size, sample_size, oracle_pct)
        (12, 250, 0.05),
        (25, 250, 0.10),
        (50, 250, 0.20),
        (125, 250, 0.50),
        (25, 500, 0.05),
        (50, 500, 0.10),
        (100, 500, 0.20),
        (250, 500, 0.50),
    ]

    print("Simulating oracle variance estimates...")
    print()

    results = []

    for oracle_size, sample_size, oracle_pct in scenarios:
        print(f"Scenario: n_oracle={oracle_size}, n_sample={sample_size} ({oracle_pct:.0%})")

        # Simulate judge scores and oracle labels
        # Use correlation structure similar to real data
        rng = np.random.RandomState(42)

        # Judge scores (somewhat noisy)
        all_judge_scores = rng.beta(2, 2, size=sample_size)

        # Oracle labels (correlated with judge but with noise)
        oracle_mask = np.zeros(sample_size, dtype=bool)
        oracle_mask[:oracle_size] = True
        rng.shuffle(oracle_mask)  # Random oracle selection

        oracle_judge_scores = all_judge_scores[oracle_mask]
        # True labels = judge + noise, keeping monotone relationship
        oracle_labels = np.clip(
            oracle_judge_scores + rng.normal(0, 0.15, size=oracle_size),
            0.0, 1.0
        )

        # Compute jackknife variance
        jk_var, K = oracle_jackknife_variance(
            oracle_judge_scores,
            oracle_labels,
            oracle_mask,
            all_judge_scores,
            random_seed=42
        )

        # Compute bootstrap variance
        boot_var = oracle_bootstrap_variance(
            oracle_judge_scores,
            oracle_labels,
            oracle_mask,
            all_judge_scores,
            n_bootstrap=50,
            random_seed=42
        )

        # Estimate "true" variance by repeating the process
        # (simulates multiple independent experimental runs)
        n_reps = 100
        rep_estimates = []
        for rep in range(n_reps):
            rep_rng = np.random.RandomState(42 + rep)

            # New oracle sample
            rep_oracle_scores = oracle_judge_scores + rep_rng.normal(0, 0.05, size=oracle_size)
            rep_oracle_labels = oracle_labels + rep_rng.normal(0, 0.05, size=oracle_size)
            rep_oracle_labels = np.clip(rep_oracle_labels, 0.0, 1.0)

            # Fit calibrator
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(rep_oracle_scores, rep_oracle_labels)

            # Estimate
            calibrated = np.clip(iso.predict(all_judge_scores), 0.0, 1.0)
            estimate = float(np.mean(calibrated))
            rep_estimates.append(estimate)

        emp_var = float(np.var(rep_estimates, ddof=1))

        # Compute ratios
        jk_ratio = jk_var / emp_var if emp_var > 0 else 0.0
        boot_ratio = boot_var / emp_var if emp_var > 0 else 0.0

        print(f"  Jackknife (K={K}): var={jk_var:.2e}, ratio={jk_ratio:.2f}")
        print(f"  Bootstrap (B=50): var={boot_var:.2e}, ratio={boot_ratio:.2f}")
        print(f"  Empirical (N=100): var={emp_var:.2e}")
        print()

        results.append(OUAComparison(
            oracle_size=oracle_size,
            sample_size=sample_size,
            oracle_pct=oracle_pct,
            jackknife_var=jk_var,
            jackknife_k=K,
            bootstrap_var=boot_var,
            bootstrap_b=50,
            empirical_var=emp_var,
            empirical_n_runs=n_reps,
            jackknife_ratio=jk_ratio,
            bootstrap_ratio=boot_ratio
        ))

    # Summary
    print("=" * 80)
    print("Summary: Which method better estimates oracle variance?")
    print("=" * 80)
    print()
    print(f"{'Oracle %':<10} {'Oracle n':<10} {'JK Ratio':<12} {'Boot Ratio':<12} {'Better':<10}")
    print("-" * 60)

    for r in results:
        better = "Bootstrap" if abs(r.bootstrap_ratio - 1.0) < abs(r.jackknife_ratio - 1.0) else "Jackknife"
        print(f"{r.oracle_pct:>7.0%}   {r.oracle_size:>8}   {r.jackknife_ratio:>10.2f}   {r.bootstrap_ratio:>10.2f}   {better:<10}")

    print()
    print("Ratio interpretation:")
    print("  < 1.0: Method underestimates variance (leads to under-coverage)")
    print("  = 1.0: Method accurately estimates variance")
    print("  > 1.0: Method overestimates variance (leads to over-coverage)")
    print()

    # Average ratios
    avg_jk = np.mean([r.jackknife_ratio for r in results])
    avg_boot = np.mean([r.bootstrap_ratio for r in results])

    print(f"Average jackknife ratio: {avg_jk:.2f}")
    print(f"Average bootstrap ratio: {avg_boot:.2f}")
    print()

    # Low oracle scenarios
    low_oracle = [r for r in results if r.oracle_pct <= 0.10]
    if low_oracle:
        avg_jk_low = np.mean([r.jackknife_ratio for r in low_oracle])
        avg_boot_low = np.mean([r.bootstrap_ratio for r in low_oracle])
        print(f"Low oracle (≤10%) - Jackknife ratio: {avg_jk_low:.2f}, Bootstrap ratio: {avg_boot_low:.2f}")


if __name__ == "__main__":
    main()
