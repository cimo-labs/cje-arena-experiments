#!/usr/bin/env python3
"""Analyze stacking weight variability with/without covariates."""

import json
import numpy as np
from pathlib import Path

# Load ablation results
results_path = Path("results/all_experiments.jsonl")

# Filter for stacked-dr with n=250, oracle_coverage=0.25
with_cov_weights = []
no_cov_weights = []
with_cov_rmse = []
no_cov_rmse = []

with open(results_path) as f:
    for line in f:
        result = json.loads(line)
        spec = result["spec"]

        if (spec["estimator"] == "stacked-dr" and
            spec["sample_size"] == 250 and
            spec["oracle_coverage"] == 0.25):

            use_cov = spec.get("extra", {}).get("use_covariates", False)
            rmse = result["rmse_vs_oracle"]

            # Extract stacking weights
            metadata = result.get("metadata", {})
            stacking_weights = metadata.get("stacking_weights", {})

            # Get weights for main 3 policies
            if stacking_weights:
                weights_list = []
                for policy in ["clone", "parallel_universe_prompt", "premium"]:
                    if policy in stacking_weights:
                        weights_list.append(stacking_weights[policy])

                if weights_list:
                    # Average weights across policies
                    avg_weights = np.mean(weights_list, axis=0)

                    if use_cov:
                        with_cov_weights.append(avg_weights)
                        with_cov_rmse.append(rmse)
                    else:
                        no_cov_weights.append(avg_weights)
                        no_cov_rmse.append(rmse)

# Convert to numpy arrays
with_cov_weights = np.array(with_cov_weights)
no_cov_weights = np.array(no_cov_weights)
with_cov_rmse = np.array(with_cov_rmse)
no_cov_rmse = np.array(no_cov_rmse)

print("="*80)
print("STACKING WEIGHT ANALYSIS")
print("="*80)

print(f"\nNUMBER OF RUNS:")
print(f"  WITH covariates: {len(with_cov_weights)}")
print(f"  WITHOUT covariates: {len(no_cov_weights)}")

# Analyze weight variability
print(f"\nWEIGHT STATISTICS (averaged across 3 main policies):")
print(f"\nWITHOUT COVARIATES:")
print(f"  Mean weights: {np.mean(no_cov_weights, axis=0)}")
print(f"  Std weights:  {np.std(no_cov_weights, axis=0)}")
print(f"  Weight entropy: {np.mean([-np.sum(w * np.log(w + 1e-10)) for w in no_cov_weights]):.3f}")

print(f"\nWITH COVARIATES:")
print(f"  Mean weights: {np.mean(with_cov_weights, axis=0)}")
print(f"  Std weights:  {np.std(with_cov_weights, axis=0)}")
print(f"  Weight entropy: {np.mean([-np.sum(w * np.log(w + 1e-10)) for w in with_cov_weights]):.3f}")

# Check correlation between weight variability and RMSE
print(f"\nCORRELATION BETWEEN WEIGHT CONCENTRATION AND RMSE:")

# Compute weight concentration (max weight)
no_cov_max_weights = np.max(no_cov_weights, axis=1)
with_cov_max_weights = np.max(with_cov_weights, axis=1)

# Compute Gini coefficient (measure of inequality)
def gini(weights):
    """Compute Gini coefficient (0=uniform, 1=all mass on one component)."""
    sorted_w = np.sort(weights)
    n = len(weights)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_w)) / (n * np.sum(sorted_w)) - (n + 1) / n

no_cov_gini = np.array([gini(w) for w in no_cov_weights])
with_cov_gini = np.array([gini(w) for w in with_cov_weights])

print(f"\n  WITHOUT cov: Gini vs RMSE correlation = {np.corrcoef(no_cov_gini, no_cov_rmse)[0,1]:.3f}")
print(f"  WITH cov: Gini vs RMSE correlation = {np.corrcoef(with_cov_gini, with_cov_rmse)[0,1]:.3f}")

print(f"\n  WITHOUT cov: Max weight vs RMSE correlation = {np.corrcoef(no_cov_max_weights, no_cov_rmse)[0,1]:.3f}")
print(f"  WITH cov: Max weight vs RMSE correlation = {np.corrcoef(with_cov_max_weights, with_cov_rmse)[0,1]:.3f}")

# Analyze weight distribution
print(f"\nWEIGHT CONCENTRATION:")
print(f"  WITHOUT cov: mean max weight = {np.mean(no_cov_max_weights):.3f}, std = {np.std(no_cov_max_weights):.3f}")
print(f"  WITH cov: mean max weight = {np.mean(with_cov_max_weights):.3f}, std = {np.std(with_cov_max_weights):.3f}")

# Check if weights are more variable with covariates
print(f"\nWEIGHT VARIABILITY ACROSS SEEDS:")
print(f"  WITHOUT cov: mean std across components = {np.mean(np.std(no_cov_weights, axis=0)):.3f}")
print(f"  WITH cov: mean std across components = {np.mean(np.std(with_cov_weights, axis=0)):.3f}")

# Find seeds with most extreme weights
print(f"\nEXTREME WEIGHT CASES:")
worst_with_cov_idx = np.argmax(with_cov_rmse)
best_with_cov_idx = np.argmin(with_cov_rmse)
worst_no_cov_idx = np.argmax(no_cov_rmse)
best_no_cov_idx = np.argmin(no_cov_rmse)

print(f"\n  WITH cov - worst RMSE ({with_cov_rmse[worst_with_cov_idx]:.4f}): weights = {with_cov_weights[worst_with_cov_idx]}")
print(f"  WITH cov - best RMSE ({with_cov_rmse[best_with_cov_idx]:.4f}): weights = {with_cov_weights[best_with_cov_idx]}")
print(f"  WITHOUT cov - worst RMSE ({no_cov_rmse[worst_no_cov_idx]:.4f}): weights = {no_cov_weights[worst_no_cov_idx]}")
print(f"  WITHOUT cov - best RMSE ({no_cov_rmse[best_no_cov_idx]:.4f}): weights = {no_cov_weights[best_no_cov_idx]}")

# Breakdown by component
print(f"\nPER-COMPONENT WEIGHT ANALYSIS:")
for i, component in enumerate(["DR-CPO", "MRDR", "TMLE"]):
    print(f"\n  {component}:")
    print(f"    WITHOUT cov: mean = {np.mean(no_cov_weights[:, i]):.3f}, std = {np.std(no_cov_weights[:, i]):.3f}")
    print(f"    WITH cov: mean = {np.mean(with_cov_weights[:, i]):.3f}, std = {np.std(with_cov_weights[:, i]):.3f}")
    print(f"    Change in mean: {(np.mean(with_cov_weights[:, i]) - np.mean(no_cov_weights[:, i])):.3f}")
    print(f"    Change in variability: {(np.std(with_cov_weights[:, i]) / np.std(no_cov_weights[:, i]) - 1) * 100:.1f}%")
