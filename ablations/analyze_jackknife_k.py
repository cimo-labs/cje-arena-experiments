"""Analyze how the number of folds K affects jackknife variance estimates."""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from typing import List, Tuple


def jackknife_variance_for_k(
    judge_scores: np.ndarray,
    oracle_labels: np.ndarray,
    all_scores: np.ndarray,
    k: int,
    random_seed: int = 42
) -> Tuple[float, List[float]]:
    """Compute jackknife variance with K folds.

    Returns:
        Tuple of (jackknife variance, list of K estimates)
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)

    jack_estimates = []

    for train_idx, test_idx in kf.split(judge_scores):
        # Train on K-1 folds (leave one out)
        train_scores = judge_scores[train_idx]
        train_labels = oracle_labels[train_idx]

        # Fit isotonic on training fold
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(train_scores, train_labels)

        # Predict on all data and compute mean
        calibrated = np.clip(iso.predict(all_scores), 0.0, 1.0)
        jack_estimate = float(np.mean(calibrated))
        jack_estimates.append(jack_estimate)

    K = len(jack_estimates)
    psi_bar = float(np.mean(jack_estimates))

    # Standard jackknife variance formula
    var_jk = (K - 1) / K * float(np.mean((np.array(jack_estimates) - psi_bar) ** 2))

    return var_jk, jack_estimates


def analyze_k_sensitivity():
    """Analyze how K affects variance estimate."""

    print("=" * 80)
    print("How does K (number of folds) affect jackknife variance?")
    print("=" * 80)
    print()

    # Simulate scenarios
    scenarios = [
        (12, 250, "Very small oracle (5%)"),
        (25, 250, "Small oracle (10%)"),
        (50, 250, "Medium oracle (20%)"),
        (125, 250, "Large oracle (50%)"),
    ]

    k_values = [2, 3, 5, 10]

    for n_oracle, n_sample, desc in scenarios:
        print(f"\n{desc}: n_oracle={n_oracle}, n_sample={n_sample}")
        print("-" * 70)

        # Generate simulated data
        rng = np.random.RandomState(42)
        all_scores = rng.beta(2, 2, size=n_sample)

        # Oracle samples (first n_oracle)
        oracle_scores = all_scores[:n_oracle]
        oracle_labels = np.clip(
            oracle_scores + rng.normal(0, 0.15, size=n_oracle),
            0.0, 1.0
        )

        # Try different K values
        results = []
        for k in k_values:
            if k > n_oracle:
                continue  # Skip if not enough samples

            var_jk, estimates = jackknife_variance_for_k(
                oracle_scores, oracle_labels, all_scores, k, random_seed=42
            )

            # Compute statistics
            train_size = int(n_oracle * (k - 1) / k)
            spread = max(estimates) - min(estimates)
            emp_std = np.std(estimates, ddof=1)
            factor = (k - 1) / k

            results.append({
                'k': k,
                'train_size': train_size,
                'n_estimates': len(estimates),
                'var_jk': var_jk,
                'spread': spread,
                'emp_std': emp_std,
                'factor': factor,
                'estimates': estimates
            })

        # Print table
        print(f"\n{'K':<4} {'Train':<6} {'Factor':<8} {'Spread':<10} {'Emp Std':<10} {'Var_JK':<12}")
        print(f"{'':4} {'size':<6} {'(K-1)/K':<8} {'max-min':<10} {'(unnorm)':<10} {'(final)':<12}")
        print("-" * 70)

        for r in results:
            print(f"{r['k']:<4} {r['train_size']:<6} {r['factor']:<8.3f} "
                  f"{r['spread']:<10.4f} {r['emp_std']:<10.4f} {r['var_jk']:<12.2e}")

        # Show raw estimates for intuition
        if n_oracle <= 50:
            print(f"\nRaw jackknife estimates by K:")
            for r in results:
                est_str = ", ".join([f"{e:.4f}" for e in r['estimates']])
                print(f"  K={r['k']}: [{est_str}]")

    # Deep dive: K=3 vs K=5 for n_oracle=12
    print("\n" + "=" * 80)
    print("DETAILED: K=3 vs K=5 for n_oracle=12 (most critical case)")
    print("=" * 80)

    n_oracle = 12
    n_sample = 250
    rng = np.random.RandomState(42)
    all_scores = rng.beta(2, 2, size=n_sample)
    oracle_scores = all_scores[:n_oracle]
    oracle_labels = np.clip(
        oracle_scores + rng.normal(0, 0.15, size=n_oracle),
        0.0, 1.0
    )

    print(f"\nOracle scores (n={n_oracle}):")
    print(f"  {oracle_scores}")
    print(f"\nOracle labels:")
    print(f"  {oracle_labels}")

    for k in [3, 5]:
        print(f"\n--- K={k} folds ---")
        var_jk, estimates = jackknife_variance_for_k(
            oracle_scores, oracle_labels, all_scores, k, random_seed=42
        )

        train_size = int(n_oracle * (k - 1) / k)
        print(f"Training size per fold: {train_size}/{n_oracle} ({train_size/n_oracle:.0%})")
        print(f"Number of jackknife estimates: {len(estimates)}")
        print(f"\nJackknife estimates:")
        for i, est in enumerate(estimates):
            print(f"  Fold {i}: {est:.6f}")

        mean_est = np.mean(estimates)
        std_est = np.std(estimates, ddof=1)
        emp_var = std_est ** 2
        factor = (k - 1) / k

        print(f"\nMean estimate: {mean_est:.6f}")
        print(f"Empirical std: {std_est:.6f}")
        print(f"Empirical var (unnormalized): {emp_var:.2e}")
        print(f"Correction factor (K-1)/K: {factor:.4f}")
        print(f"Final jackknife variance: {var_jk:.2e}")
        print(f"  = {factor:.4f} × {emp_var:.2e}")

    # Sensitivity to random seed
    print("\n" + "=" * 80)
    print("SENSITIVITY: How stable is variance estimate across random splits?")
    print("=" * 80)

    n_oracle = 12
    n_sample = 250
    rng = np.random.RandomState(42)
    all_scores = rng.beta(2, 2, size=n_sample)
    oracle_scores = all_scores[:n_oracle]
    oracle_labels = np.clip(
        oracle_scores + rng.normal(0, 0.15, size=n_oracle),
        0.0, 1.0
    )

    for k in [3, 5]:
        print(f"\n--- K={k} across different random splits ---")
        variances = []
        for seed in range(10):
            var_jk, _ = jackknife_variance_for_k(
                oracle_scores, oracle_labels, all_scores, k, random_seed=seed
            )
            variances.append(var_jk)

        print(f"Variance estimates across 10 random splits:")
        print(f"  {[f'{v:.2e}' for v in variances]}")
        print(f"  Mean: {np.mean(variances):.2e}")
        print(f"  Std:  {np.std(variances, ddof=1):.2e}")
        print(f"  CV:   {np.std(variances, ddof=1) / np.mean(variances):.2f}")


if __name__ == "__main__":
    analyze_k_sensitivity()
