#!/usr/bin/env python3
"""Generate a forest plot from a single ablation run showing estimates vs oracle truth."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
SAMPLE_SIZE = 1000
ORACLE_COVERAGE = 0.25
SEED = 0
ESTIMATOR = "direct+cov"

print("="*70)
print("FOREST PLOT FROM ABLATION DATA")
print("="*70)
print(f"\nConfiguration:")
print(f"  Sample size: {SAMPLE_SIZE}")
print(f"  Oracle coverage: {int(ORACLE_COVERAGE*100)}%")
print(f"  Seed: {SEED}")
print(f"  Estimator: {ESTIMATOR}")

# Load ablation results
results_path = Path("results/all_experiments.jsonl")
print(f"\nLoading ablation results from {results_path}...")

# Find matching experiment
matching_result = None
with open(results_path) as f:
    for line in f:
        result = json.loads(line)
        spec = result.get("spec", {})
        extra = spec.get("extra", {})

        # Match criteria
        if (result.get("success") and
            spec.get("estimator") == "direct" and
            spec.get("sample_size") == SAMPLE_SIZE and
            abs(spec.get("oracle_coverage", 0) - ORACLE_COVERAGE) < 0.01 and
            extra.get("use_covariates") == True and
            result.get("seed") == SEED):

            matching_result = result
            break

if not matching_result:
    print(f"No matching experiment found!")
    exit(1)

print(f"Found matching experiment (seed={matching_result['seed']})")

# Extract data
estimates = matching_result.get("estimates", {})
cis = matching_result.get("confidence_intervals", {})
oracles = matching_result.get("oracle_truths", {})
n_oracle = matching_result.get("oracle_slice_size", 250)

# Build policy data
policy_data = {}

# Add estimated policies
for policy in estimates:
    est = estimates[policy]
    if np.isfinite(est):
        ci_lower, ci_upper = cis.get(policy, (None, None))
        oracle = oracles.get(policy)

        policy_data[policy] = {
            "estimate": est,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "oracle_truth": oracle,
        }

# Add base policy with computed CI
if "base" in oracles:
    oracle_val = oracles["base"]
    # Compute binomial CI for oracle proportion
    se_oracle = np.sqrt(oracle_val * (1 - oracle_val) / n_oracle)
    ci_lower = oracle_val - 1.96 * se_oracle
    ci_upper = oracle_val + 1.96 * se_oracle

    policy_data["base"] = {
        "estimate": oracle_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "oracle_truth": oracle_val,
    }

# Print results
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"\n{'Policy':<30} {'Estimate':>10} {'95% CI':>20} {'Oracle':>10}")
print("-" * 80)

for policy in sorted(policy_data.keys()):
    data = policy_data[policy]
    est = data["estimate"]
    ci_l = data["ci_lower"]
    ci_u = data["ci_upper"]
    oracle = data["oracle_truth"]

    oracle_str = f"{oracle:.4f}" if oracle is not None else "N/A"
    ci_str = f"[{ci_l:.4f}, {ci_u:.4f}]" if ci_l is not None else "N/A"
    print(f"{policy:<30} {est:>10.4f}  {ci_str:>20}  {oracle_str:>10}")

# Create forest plot
print("\nCreating forest plot...")

fig, ax = plt.subplots(figsize=(10, max(4, len(policy_data) * 0.35)))

# Sort policies alphabetically
sorted_policies = sorted(policy_data.keys())
sorted_estimates = [policy_data[p]["estimate"] for p in sorted_policies]
sorted_ci_lower = [policy_data[p]["ci_lower"] for p in sorted_policies]
sorted_ci_upper = [policy_data[p]["ci_upper"] for p in sorted_policies]
sorted_oracles = [policy_data[p]["oracle_truth"] for p in sorted_policies]

# Y positions (reverse so first is at top)
y_pos = np.arange(len(sorted_policies))[::-1]

# Calculate error bars
yerr_lower = [est - ci_l if ci_l is not None else 0
              for est, ci_l in zip(sorted_estimates, sorted_ci_lower)]
yerr_upper = [ci_u - est if ci_u is not None else 0
              for est, ci_u in zip(sorted_estimates, sorted_ci_upper)]

# Plot oracle truth
oracle_x = [o for o in sorted_oracles if o is not None]
oracle_y = [y for y, o in zip(y_pos, sorted_oracles) if o is not None]

if oracle_x:
    ax.scatter(oracle_x, oracle_y, color='red', marker='d', s=100,
              label='Oracle truth', zorder=3, alpha=0.8, edgecolors='darkred', linewidths=1.5)

# Plot estimates with error bars
ax.errorbar(sorted_estimates, y_pos,
            xerr=[yerr_lower, yerr_upper],
            fmt='o', markersize=8, capsize=5, capthick=2,
            color='steelblue', ecolor='steelblue',
            label='Estimate ± 95% CI', zorder=2)

# Formatting
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_policies, fontsize=11)
ax.set_xlabel('Oracle Score', fontsize=12)
ax.set_ylabel('Policy', fontsize=12)
ax.set_title(f'Forest Plot: {ESTIMATOR}\n(n={SAMPLE_SIZE}, oracle={int(ORACLE_COVERAGE*100)}%, seed={SEED})',
             fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='best', fontsize=10)
ax.grid(axis='x', alpha=0.3, linestyle=':')

plt.tight_layout()

# Save figure
output_path = Path(f"results/analysis/forest_plot_{ESTIMATOR.replace('+', '_')}_n{SAMPLE_SIZE}_oracle{int(ORACLE_COVERAGE*100)}_seed{SEED}.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved forest plot to {output_path}")

print("\n" + "="*70)
print("COMPLETE")
print("="*70)
