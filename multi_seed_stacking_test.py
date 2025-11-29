#!/usr/bin/env python3
"""Test stacked-dr with/without covariates across multiple seeds."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent))

from cje import load_dataset_from_jsonl
from cje.calibration.dataset import calibrate_dataset
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.estimators.stacking import StackedDREstimator
from cje.data.fresh_draws import load_fresh_draws_auto, FreshDrawDataset, compute_response_covariates
from cje.data.models import Dataset

# Configuration
DATASET_PATH = "data/cje_dataset.jsonl"
N_SAMPLES = 250
ORACLE_COVERAGE = 0.25
N_SEEDS = 10  # Test with 10 seeds

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

def test_seed(seed, dataset, data_dir):
    """Test stacking with and without covariates for a single seed."""
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

    results = {}

    # WITH COVARIATES
    try:
        calibrated_with_cov, cal_result_with_cov = calibrate_dataset(
            sampled_dataset,
            random_seed=seed,
            n_folds=5,
            calibration_mode="auto",
            covariate_names=["response_length"],
            enable_cross_fit=True,
        )

        sampler_with_cov = PrecomputedSampler(calibrated_with_cov, calibrate=False)

        # Load fresh draws
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

        estimator_with_cov = StackedDREstimator(
            sampler=sampler_with_cov,
            estimators=["dr-cpo", "mrdr", "tmle"],
            n_folds=5,
            reward_calibrator=cal_result_with_cov.calibrator,
            parallel=False,
            use_calibrated_weights=True,
        )

        for policy, fresh_draw_data in fresh_draws_with_cov.items():
            estimator_with_cov.add_fresh_draws(policy, fresh_draw_data)

        result_with_cov = estimator_with_cov.estimate()
        rmse_with_cov = compute_rmse(result_with_cov.estimates, sampler_with_cov.target_policies)

        # Extract component estimates
        comp_ests_with = result_with_cov.metadata.get("component_estimates", {})

        results["with_cov_rmse"] = rmse_with_cov
        results["with_cov_weights"] = result_with_cov.metadata.get("stacking_weights", {})
        results["with_cov_components"] = comp_ests_with

        print(f"\nWITH COVARIATES: RMSE = {rmse_with_cov:.6f}")

        # Print component RMSEs
        for comp_name, comp_vals in comp_ests_with.items():
            comp_estimates = [comp_vals[p] for p in ["clone", "parallel_universe_prompt", "premium"]]
            comp_rmse = compute_rmse(comp_estimates, ["clone", "parallel_universe_prompt", "premium"])
            print(f"  {comp_name}: component RMSE = {comp_rmse:.6f}")

    except Exception as e:
        print(f"WITH COVARIATES FAILED: {e}")
        results["with_cov_rmse"] = np.nan

    # WITHOUT COVARIATES
    try:
        calibrated_no_cov, cal_result_no_cov = calibrate_dataset(
            sampled_dataset,
            random_seed=seed,
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

        estimator_no_cov = StackedDREstimator(
            sampler=sampler_no_cov,
            estimators=["dr-cpo", "mrdr", "tmle"],
            n_folds=5,
            reward_calibrator=cal_result_no_cov.calibrator,
            parallel=False,
            use_calibrated_weights=True,
        )

        for policy, fresh_draw_data in fresh_draws_no_cov.items():
            estimator_no_cov.add_fresh_draws(policy, fresh_draw_data)

        result_no_cov = estimator_no_cov.estimate()
        rmse_no_cov = compute_rmse(result_no_cov.estimates, sampler_no_cov.target_policies)

        results["no_cov_rmse"] = rmse_no_cov
        results["no_cov_weights"] = result_no_cov.metadata.get("stacking_weights", {})

        print(f"\nWITHOUT COVARIATES: RMSE = {rmse_no_cov:.6f}")

        degradation_pct = (rmse_with_cov - rmse_no_cov) / rmse_no_cov * 100
        print(f"\nDEGRADATION: {degradation_pct:+.1f}%")

    except Exception as e:
        print(f"WITHOUT COVARIATES FAILED: {e}")
        results["no_cov_rmse"] = np.nan

    return results

# Load dataset once
print("Loading dataset...")
dataset = load_dataset_from_jsonl(DATASET_PATH)
data_dir = Path("data")

# Run across multiple seeds
all_results = []
for seed in range(N_SEEDS):
    results = test_seed(seed, dataset, data_dir)
    results["seed"] = seed
    all_results.append(results)

# Summary
print("\n" + "="*80)
print("SUMMARY ACROSS SEEDS")
print("="*80)

with_cov_rmses = [r["with_cov_rmse"] for r in all_results if not np.isnan(r.get("with_cov_rmse", np.nan))]
no_cov_rmses = [r["no_cov_rmse"] for r in all_results if not np.isnan(r.get("no_cov_rmse", np.nan))]

if with_cov_rmses and no_cov_rmses:
    print(f"\nWITH COVARIATES:")
    print(f"  Mean RMSE: {np.mean(with_cov_rmses):.6f}")
    print(f"  Std RMSE:  {np.std(with_cov_rmses):.6f}")

    print(f"\nWITHOUT COVARIATES:")
    print(f"  Mean RMSE: {np.mean(no_cov_rmses):.6f}")
    print(f"  Std RMSE:  {np.std(no_cov_rmses):.6f}")

    degradations = [(with_cov - no_cov) / no_cov * 100
                    for with_cov, no_cov in zip(with_cov_rmses, no_cov_rmses)]
    print(f"\nDEGRADATION:")
    print(f"  Mean: {np.mean(degradations):+.1f}%")
    print(f"  Std:  {np.std(degradations):.1f}%")
    print(f"  Range: [{np.min(degradations):+.1f}%, {np.max(degradations):+.1f}%]")

    n_degraded = sum(1 for d in degradations if d > 0)
    print(f"\nSeeds with degradation: {n_degraded}/{len(degradations)} ({n_degraded/len(degradations)*100:.0f}%)")
