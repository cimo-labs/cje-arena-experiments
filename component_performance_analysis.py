#!/usr/bin/env python3
"""Analyze component estimator performance with/without covariates across seeds."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cje import load_dataset_from_jsonl
from cje.calibration.dataset import calibrate_dataset
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.estimators.dr_base import DRCPOEstimator
from cje.estimators.tmle import TMLEEstimator
from cje.estimators.mrdr import MRDREstimator
from cje.estimators.stacking import StackedDREstimator
from cje.data.fresh_draws import load_fresh_draws_auto, FreshDrawDataset, compute_response_covariates
from cje.data.models import Dataset

# Configuration
DATASET_PATH = "data/cje_dataset.jsonl"
N_SAMPLES = 250
ORACLE_COVERAGE = 0.25
N_SEEDS = 20  # Analyze 20 seeds

ORACLE_TRUTHS = {
    "clone": 0.7620,
    "parallel_universe_prompt": 0.7708,
    "premium": 0.7623,
}

def compute_rmse(estimates, policies):
    """Compute RMSE vs oracle truths (excluding unhelpful)."""
    errors = []
    for i, policy in enumerate(policies):
        if policy == "unhelpful":
            continue
        oracle = ORACLE_TRUTHS.get(policy)
        if oracle is not None:
            error = (estimates[i] - oracle) ** 2
            errors.append(error)
    return np.sqrt(np.mean(errors)) if errors else np.nan

def run_components(sampled_dataset, seed, use_covariates, data_dir):
    """Run individual component estimators."""
    covariate_names = ["response_length"] if use_covariates else None

    # Calibrate
    calibrated, cal_result = calibrate_dataset(
        sampled_dataset,
        random_seed=seed,
        n_folds=5,
        calibration_mode="auto",
        covariate_names=covariate_names,
        enable_cross_fit=True,
    )

    sampler = PrecomputedSampler(calibrated, calibrate=False)

    # Load fresh draws
    fresh_draws = {}
    for policy in sampled_dataset.target_policies:
        try:
            all_fresh = load_fresh_draws_auto(data_dir, policy, verbose=False)
            filtered_samples = [s for s in all_fresh.samples
                              if s.prompt_id in {s.prompt_id for s in sampled_dataset.samples}]

            if filtered_samples:
                fresh_dataset = FreshDrawDataset(
                    samples=filtered_samples,
                    target_policy=policy,
                    draws_per_prompt=10,
                )
                if use_covariates:
                    fresh_dataset = compute_response_covariates(
                        fresh_dataset,
                        covariate_names=["response_length"]
                    )
                fresh_draws[policy] = fresh_dataset
        except FileNotFoundError:
            pass

    # Run each component
    results = {}

    for name, EstimatorClass in [
        ("dr-cpo", DRCPOEstimator),
        ("mrdr", MRDREstimator),
        ("tmle", TMLEEstimator),
    ]:
        try:
            estimator = EstimatorClass(
                sampler=sampler,
                n_folds=5,
                reward_calibrator=cal_result.calibrator,
                use_calibrated_weights=True,
            )

            for policy, fresh in fresh_draws.items():
                estimator.add_fresh_draws(policy, fresh)

            result = estimator.fit_and_estimate()
            rmse = compute_rmse(result.estimates, sampler.target_policies)

            results[name] = {
                "rmse": rmse,
                "estimates": result.estimates.copy(),
            }
        except Exception as e:
            print(f"  {name} failed: {e}")
            results[name] = {"rmse": np.nan, "estimates": None}

    # Also run stacking
    try:
        stacking = StackedDREstimator(
            sampler=sampler,
            estimators=["dr-cpo", "mrdr", "tmle"],
            n_folds=5,
            reward_calibrator=cal_result.calibrator,
            parallel=False,
            use_calibrated_weights=True,
        )

        for policy, fresh in fresh_draws.items():
            stacking.add_fresh_draws(policy, fresh)

        stacking_result = stacking.estimate()
        stacking_rmse = compute_rmse(stacking_result.estimates, sampler.target_policies)

        # Get weights
        weights = stacking_result.metadata.get("stacking_weights", {})
        avg_weights = np.mean([weights[p] for p in ["clone", "parallel_universe_prompt", "premium"]], axis=0)

        results["stacking"] = {
            "rmse": stacking_rmse,
            "weights": avg_weights,
            "estimates": stacking_result.estimates.copy(),
        }
    except Exception as e:
        print(f"  stacking failed: {e}")
        results["stacking"] = {"rmse": np.nan}

    return results

# Load dataset
print("Loading dataset...")
dataset = load_dataset_from_jsonl(DATASET_PATH)
data_dir = Path("data")

all_results = []

for seed in range(N_SEEDS):
    print(f"\n{'='*80}")
    print(f"SEED {seed}")
    print('='*80)

    # Sample dataset
    rng = np.random.RandomState(seed)
    all_prompt_ids = list(set(s.prompt_id for s in dataset.samples))
    sampled_prompt_ids = set(rng.choice(all_prompt_ids, size=N_SAMPLES, replace=False))

    sampled_dataset = Dataset(
        samples=[s for s in dataset.samples if s.prompt_id in sampled_prompt_ids],
        target_policies=dataset.target_policies
    )

    # Apply oracle coverage
    rng2 = np.random.RandomState(seed)
    oracle_samples = [s for s in sampled_dataset.samples if s.oracle_label is not None]
    n_keep = max(2, int(len(oracle_samples) * ORACLE_COVERAGE))
    rng2.shuffle(oracle_samples)
    oracle_samples_keep = set(s.prompt_id for s in oracle_samples[:n_keep])

    for sample in sampled_dataset.samples:
        if sample.prompt_id not in oracle_samples_keep:
            sample.oracle_label = None

    # Run with covariates
    print("\n  WITH COVARIATES:")
    with_cov = run_components(sampled_dataset, seed, use_covariates=True, data_dir=data_dir)
    for name, res in with_cov.items():
        if name == "stacking":
            print(f"    {name}: RMSE = {res['rmse']:.4f}, weights = {res.get('weights', 'N/A')}")
        else:
            print(f"    {name}: RMSE = {res['rmse']:.4f}")

    # Run without covariates
    print("\n  WITHOUT COVARIATES:")
    no_cov = run_components(sampled_dataset, seed, use_covariates=False, data_dir=data_dir)
    for name, res in no_cov.items():
        if name == "stacking":
            print(f"    {name}: RMSE = {res['rmse']:.4f}, weights = {res.get('weights', 'N/A')}")
        else:
            print(f"    {name}: RMSE = {res['rmse']:.4f}")

    # Store results
    all_results.append({
        "seed": seed,
        "with_cov": with_cov,
        "no_cov": no_cov,
    })

# Summary analysis
print("\n" + "="*80)
print("SUMMARY ANALYSIS")
print("="*80)

# Component-level performance
for component in ["dr-cpo", "mrdr", "tmle", "stacking"]:
    with_cov_rmses = [r["with_cov"][component]["rmse"] for r in all_results
                      if not np.isnan(r["with_cov"][component]["rmse"])]
    no_cov_rmses = [r["no_cov"][component]["rmse"] for r in all_results
                    if not np.isnan(r["no_cov"][component]["rmse"])]

    print(f"\n{component.upper()}:")
    print(f"  WITH cov:    mean = {np.mean(with_cov_rmses):.4f}, std = {np.std(with_cov_rmses):.4f}, median = {np.median(with_cov_rmses):.4f}")
    print(f"  WITHOUT cov: mean = {np.mean(no_cov_rmses):.4f}, std = {np.std(no_cov_rmses):.4f}, median = {np.median(no_cov_rmses):.4f}")

    if no_cov_rmses:
        degradation = (np.mean(with_cov_rmses) - np.mean(no_cov_rmses)) / np.mean(no_cov_rmses) * 100
        print(f"  Degradation: {degradation:+.1f}%")

# Analyze which component would be best if we just picked one
print("\n" + "="*80)
print("ORACLE COMPONENT SELECTION")
print("="*80)

for use_cov in [True, False]:
    label = "WITH COVARIATES" if use_cov else "WITHOUT COVARIATES"
    key = "with_cov" if use_cov else "no_cov"

    print(f"\n{label}:")

    # For each seed, find best component
    best_counts = {"dr-cpo": 0, "mrdr": 0, "tmle": 0}
    best_avg_rmse = []

    for r in all_results:
        component_rmses = {name: r[key][name]["rmse"] for name in ["dr-cpo", "mrdr", "tmle"]}
        best_component = min(component_rmses, key=component_rmses.get)
        best_counts[best_component] += 1
        best_avg_rmse.append(component_rmses[best_component])

    print(f"  Best component frequency:")
    for name, count in best_counts.items():
        print(f"    {name}: {count}/{len(all_results)} ({count/len(all_results)*100:.0f}%)")

    print(f"  Oracle single-component RMSE: {np.mean(best_avg_rmse):.4f}")

    # Compare to stacking
    stacking_rmses = [r[key]["stacking"]["rmse"] for r in all_results
                      if not np.isnan(r[key]["stacking"]["rmse"])]
    print(f"  Stacking RMSE:                 {np.mean(stacking_rmses):.4f}")
    print(f"  Gap: {(np.mean(stacking_rmses) - np.mean(best_avg_rmse)):.4f}")

# Analyze correlation between components
print("\n" + "="*80)
print("COMPONENT CORRELATION ANALYSIS")
print("="*80)

for use_cov in [True, False]:
    label = "WITH COVARIATES" if use_cov else "WITHOUT COVARIATES"
    key = "with_cov" if use_cov else "no_cov"

    print(f"\n{label}:")

    dr_cpo_rmses = np.array([r[key]["dr-cpo"]["rmse"] for r in all_results])
    mrdr_rmses = np.array([r[key]["mrdr"]["rmse"] for r in all_results])
    tmle_rmses = np.array([r[key]["tmle"]["rmse"] for r in all_results])

    print(f"  DR-CPO vs MRDR correlation: {np.corrcoef(dr_cpo_rmses, mrdr_rmses)[0,1]:.3f}")
    print(f"  DR-CPO vs TMLE correlation: {np.corrcoef(dr_cpo_rmses, tmle_rmses)[0,1]:.3f}")
    print(f"  MRDR vs TMLE correlation:   {np.corrcoef(mrdr_rmses, tmle_rmses)[0,1]:.3f}")
