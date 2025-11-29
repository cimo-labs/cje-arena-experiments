#!/usr/bin/env python3
"""Diagnose stacking covariance matrix stability with/without covariates."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent))

from cje import load_dataset_from_jsonl
from cje.calibration.dataset import calibrate_dataset
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.estimators.dr_base import DRCPOEstimator
from cje.estimators.tmle import TMLEEstimator
from cje.estimators.mrdr import MRDREstimator
from cje.data.fresh_draws import load_fresh_draws_auto, FreshDrawDataset, compute_response_covariates
from cje.data.models import Dataset

# Configuration
DATASET_PATH = "data/cje_dataset.jsonl"
N_SAMPLES = 250
ORACLE_COVERAGE = 0.25
SEED = 42

ORACLE_TRUTHS = {
    "clone": 0.7620,
    "parallel_universe_prompt": 0.7708,
    "premium": 0.7623,
}

def analyze_covariance_matrix(Sigma, label):
    """Analyze covariance matrix properties."""
    print(f"\n{label}")
    print("=" * 80)

    # Eigendecomposition
    eigenvalues = np.linalg.eigvalsh(Sigma)
    condition_number = eigenvalues.max() / max(eigenvalues.min(), 1e-10)

    print(f"Shape: {Sigma.shape}")
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Min eigenvalue: {eigenvalues.min():.6e}")
    print(f"Max eigenvalue: {eigenvalues.max():.6e}")
    print(f"Condition number: {condition_number:.2e}")

    # Check for near-singularity
    if eigenvalues.min() < 1e-6:
        print(f"\n⚠️  Near-singular! Min eigenvalue < 1e-6")

    if condition_number > 1e8:
        print(f"\n⚠️  Very ill-conditioned! κ > 1e8")
    elif condition_number > 1e6:
        print(f"\n⚠️  Ill-conditioned! κ > 1e6")

    # Check correlation structure
    print(f"\nCovariance matrix:")
    print(Sigma)

    # Compute correlations
    stds = np.sqrt(np.diag(Sigma))
    if np.all(stds > 1e-10):
        corr_matrix = Sigma / np.outer(stds, stds)
        print(f"\nCorrelation matrix:")
        print(corr_matrix)

        # Off-diagonal correlations
        n = Sigma.shape[0]
        off_diag_corr = []
        for i in range(n):
            for j in range(i+1, n):
                off_diag_corr.append(abs(corr_matrix[i, j]))

        if off_diag_corr:
            print(f"\nOff-diagonal correlations:")
            print(f"  Mean: {np.mean(off_diag_corr):.4f}")
            print(f"  Max:  {np.max(off_diag_corr):.4f}")
            print(f"  Min:  {np.min(off_diag_corr):.4f}")

    # Determinant
    det = np.linalg.det(Sigma)
    print(f"\nDeterminant: {det:.6e}")
    if abs(det) < 1e-10:
        print(f"⚠️  Near-zero determinant - matrix is nearly singular")

    return eigenvalues, condition_number

def compute_if_covariance(results_dict, policy, estimator_names):
    """Compute influence function covariance matrix."""
    # Collect IFs
    IF_list = []
    for name in estimator_names:
        if policy in results_dict[name].influence_functions:
            IF_list.append(results_dict[name].influence_functions[policy])

    # Stack into matrix
    IF_matrix = np.column_stack(IF_list)

    # Center and compute covariance
    centered_IF = IF_matrix - IF_matrix.mean(axis=0, keepdims=True)
    Sigma = np.cov(centered_IF.T)

    return Sigma, IF_matrix, centered_IF

# Load and sample dataset
print("Loading dataset...")
dataset = load_dataset_from_jsonl(DATASET_PATH)
rng = np.random.RandomState(SEED)
all_prompt_ids = list(set(s.prompt_id for s in dataset.samples))
sampled_prompt_ids = set(rng.choice(all_prompt_ids, size=N_SAMPLES, replace=False))

sampled_dataset = Dataset(
    samples=[s for s in dataset.samples if s.prompt_id in sampled_prompt_ids],
    target_policies=dataset.target_policies
)

# Apply oracle coverage
rng2 = np.random.RandomState(SEED)
oracle_samples = [s for s in sampled_dataset.samples if s.oracle_label is not None]
n_keep = max(2, int(len(oracle_samples) * ORACLE_COVERAGE))
rng2.shuffle(oracle_samples)
oracle_samples_keep = set(s.prompt_id for s in oracle_samples[:n_keep])

for sample in sampled_dataset.samples:
    if sample.prompt_id not in oracle_samples_keep:
        sample.oracle_label = None

print(f"Oracle samples: {n_keep}/{len(oracle_samples)}")

# ============================================================================
# WITH COVARIATES
# ============================================================================
print("\n" + "="*80)
print("TESTING WITH COVARIATES")
print("="*80)

calibrated_with_cov, cal_result_with_cov = calibrate_dataset(
    sampled_dataset,
    random_seed=SEED,
    n_folds=5,
    calibration_mode="auto",
    covariate_names=["response_length"],
    enable_cross_fit=True,
)

sampler_with_cov = PrecomputedSampler(calibrated_with_cov, calibrate=False)

# Load fresh draws
data_dir = Path("data")
fresh_draws_with_cov = {}
for policy in dataset.target_policies:
    try:
        all_fresh = load_fresh_draws_auto(data_dir, policy, verbose=False)
        filtered_samples = [s for s in all_fresh.samples if s.prompt_id in sampled_prompt_ids]

        if filtered_samples:
            fresh_dataset = FreshDrawDataset(
                samples=filtered_samples,
                target_policy=policy,
                draws_per_prompt=10,
            )
            fresh_dataset = compute_response_covariates(
                fresh_dataset,
                covariate_names=["response_length"]
            )
            fresh_draws_with_cov[policy] = fresh_dataset
    except FileNotFoundError:
        pass

# Fit estimators
estimators_with_cov = {
    "DR-CPO": DRCPOEstimator(sampler_with_cov, n_folds=5, reward_calibrator=cal_result_with_cov.calibrator, use_calibrated_weights=True),
    "MRDR": MRDREstimator(sampler_with_cov, n_folds=5, reward_calibrator=cal_result_with_cov.calibrator, use_calibrated_weights=True),
    "TMLE": TMLEEstimator(sampler_with_cov, n_folds=5, reward_calibrator=cal_result_with_cov.calibrator, use_calibrated_weights=True),
}

results_with_cov = {}
for name, est in estimators_with_cov.items():
    for policy, fresh in fresh_draws_with_cov.items():
        est.add_fresh_draws(policy, fresh)
    est.fit()
    result = est.estimate()
    results_with_cov[name] = result

# ============================================================================
# WITHOUT COVARIATES
# ============================================================================
print("\n" + "="*80)
print("TESTING WITHOUT COVARIATES")
print("="*80)

calibrated_no_cov, cal_result_no_cov = calibrate_dataset(
    sampled_dataset,
    random_seed=SEED,
    n_folds=5,
    calibration_mode="auto",
    covariate_names=None,
    enable_cross_fit=True,
)

sampler_no_cov = PrecomputedSampler(calibrated_no_cov, calibrate=False)

fresh_draws_no_cov = {}
for policy in dataset.target_policies:
    try:
        all_fresh = load_fresh_draws_auto(data_dir, policy, verbose=False)
        filtered_samples = [s for s in all_fresh.samples if s.prompt_id in sampled_prompt_ids]

        if filtered_samples:
            fresh_dataset = FreshDrawDataset(
                samples=filtered_samples,
                target_policy=policy,
                draws_per_prompt=10,
            )
            fresh_draws_no_cov[policy] = fresh_dataset
    except FileNotFoundError:
        pass

estimators_no_cov = {
    "DR-CPO": DRCPOEstimator(sampler_no_cov, n_folds=5, reward_calibrator=cal_result_no_cov.calibrator, use_calibrated_weights=True),
    "MRDR": MRDREstimator(sampler_no_cov, n_folds=5, reward_calibrator=cal_result_no_cov.calibrator, use_calibrated_weights=True),
    "TMLE": TMLEEstimator(sampler_no_cov, n_folds=5, reward_calibrator=cal_result_no_cov.calibrator, use_calibrated_weights=True),
}

results_no_cov = {}
for name, est in estimators_no_cov.items():
    for policy, fresh in fresh_draws_no_cov.items():
        est.add_fresh_draws(policy, fresh)
    est.fit()
    result = est.estimate()
    results_no_cov[name] = result

# ============================================================================
# COVARIANCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("COVARIANCE MATRIX ANALYSIS")
print("="*80)

for policy in ["clone", "parallel_universe_prompt", "premium"]:
    print(f"\n\n{'='*80}")
    print(f"POLICY: {policy}")
    print('='*80)

    # Compute covariance matrices
    Sigma_with, IF_with, centered_IF_with = compute_if_covariance(
        results_with_cov, policy, ["DR-CPO", "MRDR", "TMLE"]
    )

    Sigma_no, IF_no, centered_IF_no = compute_if_covariance(
        results_no_cov, policy, ["DR-CPO", "MRDR", "TMLE"]
    )

    # Analyze both
    eig_with, cond_with = analyze_covariance_matrix(Sigma_with, "WITH COVARIATES")
    eig_no, cond_no = analyze_covariance_matrix(Sigma_no, "WITHOUT COVARIATES")

    # Comparison
    print(f"\n{'='*80}")
    print(f"COMPARISON FOR {policy}")
    print('='*80)
    print(f"\nCondition number:")
    print(f"  WITH covariates:    {cond_with:.2e}")
    print(f"  WITHOUT covariates: {cond_no:.2e}")
    print(f"  Ratio (with/without): {cond_with/cond_no:.2f}x")

    print(f"\nSample size (n):")
    print(f"  WITH covariates:    {IF_with.shape[0]}")
    print(f"  WITHOUT covariates: {IF_no.shape[0]}")

    print(f"\nNumber of estimators (K): {IF_with.shape[1]}")
    print(f"Samples per estimator:")
    print(f"  WITH covariates:    {IF_with.shape[0] / IF_with.shape[1]:.1f}")
    print(f"  WITHOUT covariates: {IF_no.shape[0] / IF_no.shape[1]:.1f}")

# ============================================================================
# KEY FINDING
# ============================================================================
print("\n\n" + "="*80)
print("KEY FINDING")
print("="*80)

# Average condition numbers across policies
cond_with_avg = []
cond_no_avg = []

for policy in ["clone", "parallel_universe_prompt", "premium"]:
    Sigma_with, _, _ = compute_if_covariance(results_with_cov, policy, ["DR-CPO", "MRDR", "TMLE"])
    Sigma_no, _, _ = compute_if_covariance(results_no_cov, policy, ["DR-CPO", "MRDR", "TMLE"])

    eig_with = np.linalg.eigvalsh(Sigma_with)
    eig_no = np.linalg.eigvalsh(Sigma_no)

    cond_with_avg.append(eig_with.max() / max(eig_with.min(), 1e-10))
    cond_no_avg.append(eig_no.max() / max(eig_no.min(), 1e-10))

avg_cond_with = np.mean(cond_with_avg)
avg_cond_no = np.mean(cond_no_avg)

print(f"\nAverage condition number across 3 main policies:")
print(f"  WITH covariates:    {avg_cond_with:.2e}")
print(f"  WITHOUT covariates: {avg_cond_no:.2e}")
print(f"  Ratio:              {avg_cond_with/avg_cond_no:.2f}x {'WORSE' if avg_cond_with > avg_cond_no else 'BETTER'}")

if avg_cond_with > 2 * avg_cond_no:
    print(f"\n→ CONFIRMED: Covariance matrix is significantly more ill-conditioned with covariates")
    print(f"  This makes stacking optimization unstable")
    print(f"  Small changes in the data → large changes in optimal weights")
elif avg_cond_with < avg_cond_no / 2:
    print(f"\n→ SURPRISE: Covariance matrix is actually BETTER conditioned with covariates")
    print(f"  The problem is NOT covariance matrix conditioning")
else:
    print(f"\n→ Condition numbers are similar with/without covariates")
    print(f"  The problem is NOT primarily about covariance matrix ill-conditioning")

# Check if the issue is actually small sample size relative to K
n_with = IF_with.shape[0]
K = IF_with.shape[1]
print(f"\nSample size consideration:")
print(f"  n = {n_with} samples")
print(f"  K = {K} estimators")
print(f"  n/K = {n_with/K:.1f} samples per estimator")

if n_with / K < 50:
    print(f"\n⚠️  Small n/K ratio ({n_with/K:.1f}) means covariance is poorly estimated")
    print(f"  With only ~{n_with/K:.0f} samples per parameter, optimization is unreliable")
    print(f"  This creates high variance in weight selection, not necessarily bias")
