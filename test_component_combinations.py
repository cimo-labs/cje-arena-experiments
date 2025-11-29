#!/usr/bin/env python3
"""Test which component estimator causes stacked-dr degradation with covariates.

Uses the existing arena dataset and infrastructure.
"""

import sys
from pathlib import Path
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent))

from cje import load_dataset_from_jsonl
from cje.calibration.dataset import calibrate_dataset
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.estimators.stacking import StackedDREstimator
from cje.data.fresh_draws import load_fresh_draws_auto

# Configuration
DATASET_PATH = "data/cje_dataset.jsonl"
N_SAMPLES = 250
ORACLE_COVERAGE = 0.25
SEED = 42

# Oracle truths
ORACLE_TRUTHS = {
    "clone": 0.7620,
    "parallel_universe_prompt": 0.7708,
    "premium": 0.7623,
    "unhelpful": 0.1426
}

def compute_rmse(estimates, policies):
    """Compute RMSE vs oracle truths.

    Note: Excludes 'unhelpful' policy from RMSE calculation to match ablation
    methodology. The 'unhelpful' policy has a very different reward distribution
    (mean ~0.14 vs ~0.76 for other policies), which causes systematic calibration
    bias and dominates the error metric.
    """
    errors = []
    for i, policy in enumerate(policies):
        # Skip unhelpful policy like ablations do (base.py:314-315)
        if policy == "unhelpful":
            continue

        oracle = ORACLE_TRUTHS.get(policy)
        if oracle is not None:
            error = (estimates[i] - oracle) ** 2
            errors.append(error)
    return np.sqrt(np.mean(errors)) if errors else np.nan

def test_configuration(sampler, calibrator, fresh_draws, estimators, use_covariates, label):
    """Test a specific estimator configuration."""
    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"Components: {estimators}")
    print(f"Covariates: {'YES' if use_covariates else 'NO'}")
    print('='*80)

    try:
        estimator = StackedDREstimator(
            sampler=sampler,
            estimators=estimators,
            n_folds=5,
            reward_calibrator=calibrator,
            parallel=False,
            use_calibrated_weights=True,
        )

        # Add fresh draws
        for policy, fresh_draw_data in fresh_draws.items():
            estimator.add_fresh_draws(policy, fresh_draw_data)

        result = estimator.estimate()

        # Print estimates
        print("\nEstimates:")
        policies = sampler.target_policies
        for i, policy in enumerate(policies):
            oracle = ORACLE_TRUTHS.get(policy, np.nan)
            error = abs(result.estimates[i] - oracle)
            print(f"  {policy:30s}: {result.estimates[i]:.4f} (oracle={oracle:.4f}, error={error:.4f})")

        rmse = compute_rmse(result.estimates, policies)
        print(f"\nRMSE vs oracle: {rmse:.6f}")

        # Print stacking weights if available
        if hasattr(result, 'metadata') and result.metadata and 'stacking_weights' in result.metadata:
            print(f"\nStacking weights:")
            for policy in policies:
                if policy in result.metadata['stacking_weights']:
                    weights = result.metadata['stacking_weights'][policy]
                    print(f"  {policy:30s}: {weights}")

        # Print component estimates if available
        if hasattr(result, 'metadata') and result.metadata and 'component_estimates' in result.metadata:
            print(f"\nComponent estimates:")
            for est_name, est_vals in result.metadata['component_estimates'].items():
                print(f"  {est_name}:")
                for policy in policies:
                    if policy in est_vals:
                        oracle = ORACLE_TRUTHS.get(policy, np.nan)
                        error = abs(est_vals[policy] - oracle)
                        print(f"    {policy:28s}: {est_vals[policy]:.4f} (oracle={oracle:.4f}, error={error:.4f})")

        return rmse, result

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return np.nan, None

def main():
    print("="*80)
    print("COMPONENT COMBINATION TEST")
    print("Testing stacked-dr with/without covariates and different components")
    print("="*80)

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset_from_jsonl(DATASET_PATH)
    print(f"Loaded {len(dataset.samples)} total samples")

    # Sample subset
    rng = np.random.RandomState(SEED)
    all_prompt_ids = list(set(s.prompt_id for s in dataset.samples))
    sampled_prompt_ids = set(rng.choice(all_prompt_ids, size=N_SAMPLES, replace=False))

    from cje.data.models import Dataset
    sampled_dataset = Dataset(
        samples=[s for s in dataset.samples if s.prompt_id in sampled_prompt_ids],
        target_policies=dataset.target_policies
    )
    print(f"Sampled {len(sampled_dataset.samples)} samples")

    # Apply oracle coverage
    print(f"\nApplying oracle coverage ({ORACLE_COVERAGE})...")
    rng2 = np.random.RandomState(SEED)
    oracle_samples = [s for s in sampled_dataset.samples if s.oracle_label is not None]
    n_keep = max(2, int(len(oracle_samples) * ORACLE_COVERAGE))
    rng2.shuffle(oracle_samples)
    oracle_samples_keep = set(s.prompt_id for s in oracle_samples[:n_keep])

    for sample in sampled_dataset.samples:
        if sample.prompt_id not in oracle_samples_keep:
            sample.oracle_label = None
    print(f"  Kept {n_keep}/{len(oracle_samples)} oracle labels")

    # Test WITH covariates
    print("\n" + "="*80)
    print("PART 1: WITH COVARIATES")
    print("="*80)

    print("\nCalibrating with covariates...")
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
    print("\nLoading fresh draws...")
    data_dir = Path("data")
    fresh_draws_with_cov = {}
    for policy in dataset.target_policies:
        try:
            all_fresh = load_fresh_draws_auto(data_dir, policy, verbose=False)

            # Filter to sampled prompts
            from cje.data.fresh_draws import FreshDrawDataset, compute_response_covariates
            filtered_samples = [s for s in all_fresh.samples if s.prompt_id in sampled_prompt_ids]

            if filtered_samples:
                fresh_dataset = FreshDrawDataset(
                    samples=filtered_samples,
                    target_policy=policy,
                    draws_per_prompt=10,
                )
                # Compute covariates
                fresh_dataset = compute_response_covariates(
                    fresh_dataset,
                    covariate_names=["response_length"]
                )
                fresh_draws_with_cov[policy] = fresh_dataset
        except FileNotFoundError:
            print(f"  Warning: No fresh draws for {policy}")

    print(f"✓ Loaded fresh draws for {len(fresh_draws_with_cov)} policies")

    # Run tests with covariates
    results_with_cov = {}

    results_with_cov['all_3'] = test_configuration(
        sampler_with_cov, cal_result_with_cov.calibrator, fresh_draws_with_cov,
        ["dr-cpo", "mrdr", "tmle"], True,
        "WITH COVARIATES: All 3 components"
    )

    results_with_cov['no_mrdr'] = test_configuration(
        sampler_with_cov, cal_result_with_cov.calibrator, fresh_draws_with_cov,
        ["dr-cpo", "tmle"], True,
        "WITH COVARIATES: No MRDR (dr-cpo + tmle)"
    )

    results_with_cov['no_tmle'] = test_configuration(
        sampler_with_cov, cal_result_with_cov.calibrator, fresh_draws_with_cov,
        ["dr-cpo", "mrdr"], True,
        "WITH COVARIATES: No TMLE (dr-cpo + mrdr)"
    )

    # Test WITHOUT covariates
    print("\n" + "="*80)
    print("PART 2: WITHOUT COVARIATES")
    print("="*80)

    print("\nCalibrating without covariates...")
    calibrated_no_cov, cal_result_no_cov = calibrate_dataset(
        sampled_dataset,
        random_seed=SEED,
        n_folds=5,
        calibration_mode="auto",
        covariate_names=None,  # NO COVARIATES
        enable_cross_fit=True,
    )

    sampler_no_cov = PrecomputedSampler(calibrated_no_cov, calibrate=False)

    # Load fresh draws (no covariates)
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

    # Run tests without covariates
    results_no_cov = {}

    results_no_cov['all_3'] = test_configuration(
        sampler_no_cov, cal_result_no_cov.calibrator, fresh_draws_no_cov,
        ["dr-cpo", "mrdr", "tmle"], False,
        "WITHOUT COVARIATES: All 3 components"
    )

    # Summary
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)

    with_cov_all = results_with_cov['all_3'][0]
    with_cov_no_mrdr = results_with_cov['no_mrdr'][0]
    with_cov_no_tmle = results_with_cov['no_tmle'][0]
    without_cov_all = results_no_cov['all_3'][0]

    print(f"\nWITH COVARIATES:")
    print(f"  All 3 components:     {with_cov_all:.6f}")
    print(f"  Without MRDR:         {with_cov_no_mrdr:.6f}  ({(with_cov_no_mrdr-with_cov_all)/with_cov_all*100:+.1f}%)")
    print(f"  Without TMLE:         {with_cov_no_tmle:.6f}  ({(with_cov_no_tmle-with_cov_all)/with_cov_all*100:+.1f}%)")

    print(f"\nWITHOUT COVARIATES:")
    print(f"  All 3 components:     {without_cov_all:.6f}")

    print(f"\nDEGRADATION FROM ADDING COVARIATES:")
    degradation = (with_cov_all - without_cov_all) / without_cov_all * 100
    print(f"  {degradation:+.1f}%")

    if with_cov_no_mrdr < with_cov_all:
        improvement = (1 - with_cov_no_mrdr / with_cov_all) * 100
        print(f"\n✓ CONFIRMED: Removing MRDR improves by {improvement:.1f}%")
        print(f"  → MRDR is the culprit!")

    if with_cov_no_tmle < with_cov_all:
        improvement = (1 - with_cov_no_tmle / with_cov_all) * 100
        print(f"\n✓ CONFIRMED: Removing TMLE improves by {improvement:.1f}%")
        print(f"  → TMLE is the culprit!")

    if with_cov_no_mrdr >= with_cov_all and with_cov_no_tmle >= with_cov_all:
        print(f"\n✗ Neither removal improves performance.")
        print(f"  The issue may be in the interaction between components.")

if __name__ == "__main__":
    main()
