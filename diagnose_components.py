#!/usr/bin/env python3
"""Diagnose why MRDR gets zero weight with covariates."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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

# Calibrate with covariates
print("Calibrating with covariates...")
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
print("Loading fresh draws...")
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

# Fit individual estimators
print("\n" + "="*80)
print("COMPONENT ESTIMATES WITH COVARIATES (main 3 policies only)")
print("="*80)

estimators_dict = {
    "DR-CPO": DRCPOEstimator(sampler_with_cov, n_folds=5, reward_calibrator=cal_result_with_cov.calibrator, use_calibrated_weights=True),
    "MRDR": MRDREstimator(sampler_with_cov, n_folds=5, reward_calibrator=cal_result_with_cov.calibrator, use_calibrated_weights=True),
    "TMLE": TMLEEstimator(sampler_with_cov, n_folds=5, reward_calibrator=cal_result_with_cov.calibrator, use_calibrated_weights=True),
}

results = {}
for name, est in estimators_dict.items():
    for policy, fresh in fresh_draws_with_cov.items():
        est.add_fresh_draws(policy, fresh)
    est.fit()  # Must fit before estimate
    result = est.estimate()
    results[name] = result

# Print estimates
print("\nEstimates:")
print(f"{'Policy':<30s} {'DR-CPO':<10s} {'MRDR':<10s} {'TMLE':<10s} {'Oracle':<10s}")
print("-" * 80)
for i, policy in enumerate(dataset.target_policies):
    if policy in ORACLE_TRUTHS:
        oracle = ORACLE_TRUTHS[policy]
        dr_cpo = results["DR-CPO"].estimates[i]
        mrdr = results["MRDR"].estimates[i]
        tmle = results["TMLE"].estimates[i]
        print(f"{policy:<30s} {dr_cpo:<10.4f} {mrdr:<10.4f} {tmle:<10.4f} {oracle:<10.4f}")

# Print errors
print("\nAbsolute Errors:")
print(f"{'Policy':<30s} {'DR-CPO':<10s} {'MRDR':<10s} {'TMLE':<10s}")
print("-" * 80)
for i, policy in enumerate(dataset.target_policies):
    if policy in ORACLE_TRUTHS:
        oracle = ORACLE_TRUTHS[policy]
        dr_cpo_err = abs(results["DR-CPO"].estimates[i] - oracle)
        mrdr_err = abs(results["MRDR"].estimates[i] - oracle)
        tmle_err = abs(results["TMLE"].estimates[i] - oracle)
        print(f"{policy:<30s} {dr_cpo_err:<10.4f} {mrdr_err:<10.4f} {tmle_err:<10.4f}")

# Compute average errors
avg_errors = {}
for name in ["DR-CPO", "MRDR", "TMLE"]:
    errors = [abs(results[name].estimates[i] - ORACLE_TRUTHS[policy])
              for i, policy in enumerate(dataset.target_policies)
              if policy in ORACLE_TRUTHS]
    avg_errors[name] = np.mean(errors)

print(f"\nAverage Absolute Error:")
for name, err in avg_errors.items():
    print(f"  {name:<10s}: {err:.6f}")

# Check influence function variance
print("\n" + "="*80)
print("INFLUENCE FUNCTION VARIANCE (main 3 policies)")
print("="*80)
print(f"{'Policy':<30s} {'DR-CPO':<12s} {'MRDR':<12s} {'TMLE':<12s}")
print("-" * 80)
for i, policy in enumerate(dataset.target_policies):
    if policy in ORACLE_TRUTHS:
        dr_cpo_var = np.var(results["DR-CPO"].influence_functions[policy])
        mrdr_var = np.var(results["MRDR"].influence_functions[policy])
        tmle_var = np.var(results["TMLE"].influence_functions[policy])
        print(f"{policy:<30s} {dr_cpo_var:<12.6f} {mrdr_var:<12.6f} {tmle_var:<12.6f}")

# Compute correlation between influence functions
print("\n" + "="*80)
print("INFLUENCE FUNCTION CORRELATIONS (avg across main 3 policies)")
print("="*80)
correlations = {("DR-CPO", "MRDR"): [], ("DR-CPO", "TMLE"): [], ("MRDR", "TMLE"): []}
for policy in ["clone", "parallel_universe_prompt", "premium"]:
    for (name1, name2) in correlations.keys():
        if1 = results[name1].influence_functions[policy]
        if2 = results[name2].influence_functions[policy]
        corr = np.corrcoef(if1, if2)[0, 1]
        correlations[(name1, name2)].append(corr)

for (name1, name2), corrs in correlations.items():
    avg_corr = np.mean(corrs)
    print(f"  {name1} vs {name2}: {avg_corr:.4f}")

# Check TMLE epsilon values
print("\n" + "="*80)
print("TMLE TARGETING DIAGNOSTICS")
print("="*80)
if hasattr(results["TMLE"], 'metadata') and 'targeting' in results["TMLE"].metadata:
    targeting_info = results["TMLE"].metadata['targeting']
    print("\nEpsilon values (targeting parameter):")
    for policy in ["clone", "parallel_universe_prompt", "premium"]:
        if policy in targeting_info:
            info = targeting_info[policy]
            eps = info.get('epsilon', 0)
            converged = info.get('converged', False)
            iters = info.get('iters', 0)
            print(f"  {policy:<30s}: ε={eps:+.6f}, converged={converged}, iters={iters}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

# Check if MRDR has much higher variance
mrdr_avg_var = np.mean([np.var(results["MRDR"].influence_functions[p]) for p in ["clone", "parallel_universe_prompt", "premium"]])
tmle_avg_var = np.mean([np.var(results["TMLE"].influence_functions[p]) for p in ["clone", "parallel_universe_prompt", "premium"]])
dr_cpo_avg_var = np.mean([np.var(results["DR-CPO"].influence_functions[p]) for p in ["clone", "parallel_universe_prompt", "premium"]])

print(f"\nAverage IF variance:")
print(f"  DR-CPO: {dr_cpo_avg_var:.6f}")
print(f"  MRDR:   {mrdr_avg_var:.6f}  ({mrdr_avg_var/dr_cpo_avg_var:.2f}x DR-CPO)")
print(f"  TMLE:   {tmle_avg_var:.6f}  ({tmle_avg_var/dr_cpo_avg_var:.2f}x DR-CPO)")

if mrdr_avg_var > 1.5 * tmle_avg_var:
    print("\n→ MRDR has much higher IF variance than TMLE")
    print("  Stacking optimizes for minimum variance, so it downweights MRDR")
elif avg_errors["MRDR"] > 1.5 * avg_errors["TMLE"]:
    print("\n→ MRDR has much higher bias than TMLE")
    print("  Stacking optimizes for low MSE = bias² + variance, so it downweights MRDR")
else:
    print("\n→ MRDR and TMLE have similar performance")
    print("  Stacking may be choosing based on subtle differences or numerical issues")

# Now test WITHOUT covariates for comparison
print("\n" + "="*80)
print("REPEAT WITHOUT COVARIATES")
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

# Load fresh draws without covariates
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

print("\nAverage Absolute Error (WITHOUT covariates):")
for name in ["DR-CPO", "MRDR", "TMLE"]:
    errors = [abs(results_no_cov[name].estimates[i] - ORACLE_TRUTHS[policy])
              for i, policy in enumerate(dataset.target_policies)
              if policy in ORACLE_TRUTHS]
    avg_err = np.mean(errors)
    print(f"  {name:<10s}: {avg_err:.6f}")

print("\nMRDR vs TMLE correlation (WITHOUT covariates):")
corrs_no_cov = []
for policy in ["clone", "parallel_universe_prompt", "premium"]:
    if1 = results_no_cov["MRDR"].influence_functions[policy]
    if2 = results_no_cov["TMLE"].influence_functions[policy]
    corr = np.corrcoef(if1, if2)[0, 1]
    corrs_no_cov.append(corr)
print(f"  {np.mean(corrs_no_cov):.4f}")

print("\n" + "="*80)
print("KEY FINDING")
print("="*80)

corr_with = np.mean([np.corrcoef(results["MRDR"].influence_functions[p], results["TMLE"].influence_functions[p])[0,1]
                      for p in ["clone", "parallel_universe_prompt", "premium"]])
corr_without = np.mean(corrs_no_cov)

print(f"\nMRDR vs TMLE correlation:")
print(f"  WITH covariates:    {corr_with:.4f}")
print(f"  WITHOUT covariates: {corr_without:.4f}")
print(f"  Difference:         {corr_with - corr_without:+.4f}")

if corr_with > 0.98 and corr_without < 0.98:
    print("\n→ WITH covariates, MRDR and TMLE become nearly identical (>98% corr)")
    print("  Stacking sees them as redundant and picks one (TMLE)")
    print("  This creates a fragile 1-estimator ensemble instead of diversified 3-estimator")
    print("\n→ WITHOUT covariates, they're more distinct")
    print("  Stacking can diversify across all 3 estimators")
