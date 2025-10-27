#!/usr/bin/env python3
"""Run direct mode on a single scenario to get real CIs for forest plot."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List

from cje.interface import analyze_dataset
from cje.data.models import Dataset, Sample


def load_scenario_from_ablations(
    sample_size: int = 1000,
    oracle_coverage: float = 0.25,
    seed: int = 0,
) -> tuple:
    """Load a specific scenario from ablation results.

    Returns:
        (dataset, target_policies, oracle_truths)
    """
    # Load ablation results to get the scenario spec
    results_path = Path("results/all_experiments.jsonl")

    # Find matching experiment
    with open(results_path) as f:
        for line in f:
            result = json.loads(line)
            if (result.get("success") and
                result["spec"]["estimator"] == "direct" and
                result["spec"]["sample_size"] == sample_size and
                abs(result["spec"]["oracle_coverage"] - oracle_coverage) < 0.01 and
                result["spec"].get("extra", {}).get("use_covariates") and
                result["seed"] == seed):

                print(f"Found matching ablation run:")
                print(f"  Seed: {result['seed']}")
                print(f"  Sample size: {result['spec']['sample_size']}")
                print(f"  Oracle coverage: {result['spec']['oracle_coverage']}")
                print(f"  Dataset: {result['spec']['dataset_path']}")

                dataset_path = result['spec']['dataset_path']
                oracle_truths = result.get("oracle_truths", {})

                return dataset_path, oracle_truths

    raise ValueError(f"No matching experiment found for n={sample_size}, oracle={oracle_coverage}, seed={seed}")


def run_direct_mode_analysis(
    fresh_draws_dir: str,
    calibration_data_path: str = None,
    use_covariates: bool = True,
) -> Dict:
    """Run direct mode analysis and return results."""

    print("\n" + "="*70)
    print("RUNNING DIRECT MODE ANALYSIS")
    print("="*70)

    print(f"\nConfiguration:")
    print(f"  Fresh draws directory: {fresh_draws_dir}")
    print(f"  Calibration data: {calibration_data_path}")
    print(f"  Use covariates: {use_covariates}")

    # Run analysis
    result = analyze_dataset(
        fresh_draws_dir=fresh_draws_dir,
        calibration_data_path=calibration_data_path,
        estimator="direct",
        use_covariates=use_covariates,
        verbose=True,
    )

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    # Extract results
    policies = result.metadata.get("target_policies", [])

    results_dict = {
        "estimates": {},
        "confidence_intervals": {},
        "standard_errors": {},
    }

    for i, policy in enumerate(policies):
        est = result.estimates[i]
        se = result.standard_errors[i]
        ci = result.confidence_interval(alpha=0.05)
        ci_lower, ci_upper = ci[i]

        results_dict["estimates"][policy] = float(est)
        results_dict["standard_errors"][policy] = float(se)
        results_dict["confidence_intervals"][policy] = (float(ci_lower), float(ci_upper))

        print(f"\n{policy}:")
        print(f"  Estimate: {est:.4f}")
        print(f"  SE: {se:.4f}")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Print diagnostics
    print(f"\nDiagnostics:")
    print(result.diagnostics.summary())

    return results_dict


def main():
    """Main function."""

    # Configuration
    SAMPLE_SIZE = 1000
    ORACLE_COVERAGE = 0.25
    SEED = 0

    print("="*70)
    print("DIRECT MODE EXAMPLE FOR FOREST PLOT")
    print("="*70)

    # Check if we have fresh draws directory from ablations
    fresh_draws_base = Path("../data/fresh_draws")

    if not fresh_draws_base.exists():
        print(f"\nError: Fresh draws directory not found: {fresh_draws_base}")
        print("This script requires the fresh draws data from the ablation experiments.")
        return

    # Find the specific scenario
    print(f"\nLooking for scenario: n={SAMPLE_SIZE}, oracle={ORACLE_COVERAGE*100}%, seed={SEED}")

    try:
        dataset_path, oracle_truths = load_scenario_from_ablations(
            SAMPLE_SIZE, ORACLE_COVERAGE, SEED
        )
    except ValueError as e:
        print(f"\nError: {e}")
        return

    # Run the analysis
    # Note: We need to point to the fresh draws directory
    # The ablations likely used a specific dataset split

    print("\nTo run this properly, we need:")
    print("1. The fresh draws directory for each policy")
    print("2. The calibration dataset (or use oracle labels from fresh draws)")
    print("\nThe ablation experiments store these in:")
    print(f"  Dataset: {dataset_path}")
    print(f"  Fresh draws: (need to check ablation setup)")

    print("\nFor a complete example, we'd need to:")
    print("1. Extract or regenerate the fresh draws for this scenario")
    print("2. Run analyze_dataset() with proper paths")
    print("3. Compare to oracle truths:")
    for policy, truth in oracle_truths.items():
        print(f"     {policy}: {truth:.4f}")


if __name__ == "__main__":
    main()
