#!/usr/bin/env python
"""Analyze transportability with simple unbiasedness tests.

Tests whether a judge→oracle calibrator trained on base policy can transport
to target policies. Uses simple unbiasedness test: Is mean residual significantly
different from 0? Includes Bonferroni correction for multiple comparisons.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Any

from cje import load_dataset_from_jsonl
from cje.calibration import calibrate_dataset
from cje.diagnostics.transport import audit_transportability, TransportDiagnostics
from cje.data.fresh_draws import load_fresh_draws_auto
from cje.data.models import Sample

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
DATASET_PATH = DATA_DIR / "cje_dataset.jsonl"
FRESH_DRAWS_DIR = DATA_DIR / "fresh_draws"

# Experiment parameters
SUBSAMPLE_SIZE = None  # Use all base policy data
ORACLE_COVERAGE = 0.25  # 25% oracle coverage
PROBE_SIZE = None  # Use all available fresh draws per policy
SEED = 42

print("=" * 80)
print("TRANSPORTABILITY ANALYSIS WITH STATISTICAL TESTS")
print("=" * 80)

# Set seed
np.random.seed(SEED)

# ========== Step 1: Load and prepare base policy data ==========
print("\n[1/6] Loading base policy data...")
dataset = load_dataset_from_jsonl(str(DATASET_PATH))
print(f"      Loaded {dataset.n_samples} base policy samples")

# Subsample if needed
if SUBSAMPLE_SIZE and SUBSAMPLE_SIZE < len(dataset.samples):
    indices = np.random.choice(len(dataset.samples), SUBSAMPLE_SIZE, replace=False)
    dataset.samples = [dataset.samples[i] for i in sorted(indices)]
    print(f"      Subsampled to {len(dataset.samples)} samples")

# Mask oracle labels to simulate partial coverage
n_with_oracle = sum(1 for s in dataset.samples if s.oracle_label is not None)
n_keep = int(n_with_oracle * ORACLE_COVERAGE)
oracle_indices = [i for i, s in enumerate(dataset.samples) if s.oracle_label is not None]
keep_indices = set(np.random.choice(oracle_indices, n_keep, replace=False))

for i, sample in enumerate(dataset.samples):
    if i not in keep_indices and sample.oracle_label is not None:
        sample.oracle_label = None

print(f"      Kept oracle labels for {n_keep}/{n_with_oracle} samples ({ORACLE_COVERAGE:.0%})")

# ========== Step 2: Fit calibrator on base policy ==========
print("\n[2/6] Fitting calibrator on base policy...")
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    judge_field="judge_score",
    oracle_field="oracle_label",
    calibration_mode="two_stage",  # Use two-stage calibration (g(S) → isotonic)
    use_response_length=False,  # No covariates, just judge score S
    enable_cross_fit=True,
    n_folds=5,
    random_seed=SEED,
)

calibrator = cal_result.calibrator
mode = calibrator.selected_mode if hasattr(calibrator, 'selected_mode') else 'unknown'
print(f"      ✓ Fitted calibrator (mode: {mode})")
print(f"      ✓ Calibration RMSE: {cal_result.calibration_rmse:.4f}")

# ========== Step 3: Load fresh draws for target policies ==========
print("\n[3/6] Loading fresh draws for target policies...")
target_policies = ["clone", "parallel_universe_prompt", "premium", "unhelpful"]

fresh_draws_by_policy = {}
for policy in target_policies:
    try:
        fresh_draws = load_fresh_draws_auto(DATA_DIR, policy, verbose=False)

        # Sample PROBE_SIZE draws if specified
        if PROBE_SIZE and len(fresh_draws.samples) > PROBE_SIZE:
            indices = np.random.choice(len(fresh_draws.samples), PROBE_SIZE, replace=False)
            fresh_draws.samples = [fresh_draws.samples[i] for i in sorted(indices)]

        # Convert to Sample objects
        probe_samples = []
        for fd_sample in fresh_draws.samples:
            sample = Sample(
                prompt_id=fd_sample.prompt_id,
                prompt=f"prompt_{fd_sample.prompt_id}",
                response=fd_sample.response if fd_sample.response else "",
                base_policy="base",
                base_policy_logprob=0.0,
                target_policy_logprobs={policy: 0.0},
                judge_score=fd_sample.judge_score,
                oracle_label=fd_sample.oracle_label,
                metadata={
                    "judge_score": fd_sample.judge_score,
                    "oracle_label": fd_sample.oracle_label,
                }
            )
            probe_samples.append(sample)

        fresh_draws_by_policy[policy] = probe_samples
        print(f"      ✓ {policy}: {len(probe_samples)} probe samples")
    except FileNotFoundError:
        print(f"      ✗ {policy}: No fresh draws found")
        continue

# ========== Step 4: Run transportability audits with statistical tests ==========
print("\n[4/6] Running transportability audits with statistical tests...")
print("=" * 80)

results: Dict[str, TransportDiagnostics] = {}

for policy in target_policies:
    if policy not in fresh_draws_by_policy:
        continue

    probe_samples = fresh_draws_by_policy[policy]

    # Run simple unbiasedness test
    diag = audit_transportability(
        calibrator=calibrator,
        probe_samples=probe_samples,
        bins=10,
        group_label=f"policy:{policy}"
    )

    results[policy] = diag

    # Print detailed summary
    print(f"\n{policy.upper()}")
    print("-" * 80)
    print(f"  Status: {diag.status}")
    print(f"  Mean residual δ̂: {diag.delta_hat:+.4f} (95% CI: [{diag.delta_ci[0]:+.4f}, {diag.delta_ci[1]:+.4f}])")

    zero_in_ci = diag.delta_ci[0] <= 0 <= diag.delta_ci[1]
    print(f"  Unbiased (0 ∈ CI)? {'✓ Yes' if zero_in_ci else '✗ No'}")

    print(f"  Coverage: {diag.coverage:.1%}")

    if diag.status != "PASS":
        print(f"  Recommended action: {diag.recommended_action}")

# ========== Step 5: Create simple visualization ==========
print("\n[5/6] Creating visualization...")

# Bonferroni correction for multiple testing
n_policies = len(results)
alpha_single = 0.05
alpha_bonferroni = alpha_single / n_policies
z_single = stats.norm.ppf(1 - alpha_single / 2)
z_bonf = stats.norm.ppf(1 - alpha_bonferroni / 2)

# Create figure with one subplot per policy
fig, axes = plt.subplots(1, n_policies, figsize=(4 * n_policies, 4.5), sharey=True)
if n_policies == 1:
    axes = [axes]

status_colors = {
    "PASS": "#2ecc71",
    "WARN": "#f39c12",
    "FAIL": "#e74c3c",
}

for idx, (policy, diag) in enumerate(results.items()):
    ax = axes[idx]

    # Get data for this policy
    probe_samples = fresh_draws_by_policy[policy]
    judge_scores = [s.judge_score for s in probe_samples]
    oracle_labels = [s.oracle_label for s in probe_samples]

    # Compute residuals
    S = np.array(judge_scores)
    Y = np.array(oracle_labels)
    Y_pred = calibrator.predict(S)
    residuals = Y - Y_pred

    # Get status color
    status_color = status_colors.get(diag.status, "#95a5a6")

    # Bonferroni-corrected CI band
    ci_bonf_lower = diag.delta_hat - z_bonf * diag.delta_se
    ci_bonf_upper = diag.delta_hat + z_bonf * diag.delta_se
    ax.axhspan(ci_bonf_lower, ci_bonf_upper, alpha=0.4, color=status_color,
               label=f'95% CI (Bonferroni)', zorder=1, linewidth=0)

    # Scatter plot of residuals - translucent gray points
    ax.scatter(S, residuals, alpha=0.15, s=8, color='gray',
               edgecolors='none', zorder=3, label='Individual residuals')

    # Zero line (perfect calibration)
    ax.axhline(0, color='black', linestyle='--', linewidth=2, label='Perfect calibration', zorder=8, alpha=0.4)

    # Mean residual - solid line colored by status
    ax.axhline(diag.delta_hat, color=status_color, linestyle='-', linewidth=3.5,
               label=f'Mean: {diag.delta_hat:+.3f}', zorder=12, alpha=1.0)

    # Formatting
    ax.set_xlabel('Judge Score (S)', fontsize=13, fontweight='bold')
    if idx == 0:
        ax.set_ylabel('Residual (Y - f̂(S))', fontsize=13, fontweight='bold')

    # Title with status
    policy_name = policy.replace('_', ' ').title()
    ax.set_title(f"{policy_name}\n{diag.status} | n={diag.n_probe}",
                 fontsize=14, fontweight='bold', color=status_color, pad=15)

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.set_xlim(-0.05, 1.05)

    # Add status annotation
    zero_in_bonf = ci_bonf_lower <= 0 <= ci_bonf_upper
    status_text = "Unbiased ✓" if zero_in_bonf else "Biased ✗"
    ax.text(0.02, 0.98, status_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Overall title
# Get actual probe size from first policy
actual_probe_size = len(fresh_draws_by_policy[list(fresh_draws_by_policy.keys())[0]]) if fresh_draws_by_policy else PROBE_SIZE
probe_size_str = str(actual_probe_size) if actual_probe_size else "all"
fig.suptitle(
    f"Transportability Analysis: Calibrator Unbiasedness Test\n"
    f"Testing H₀: E[Y - f̂(S)] = 0 for each policy (Bonferroni-corrected α={alpha_bonferroni:.4f} for {n_policies} tests)\n"
    f"Data: oracle={ORACLE_COVERAGE:.0%}, n={probe_size_str}/policy, mode={mode}",
    fontsize=14,
    fontweight="bold",
    y=1.02
)

plt.tight_layout()

# Save figure
output_path = Path(__file__).parent / "reporting" / "transportability_statistical.png"
output_path.parent.mkdir(exist_ok=True)
fig.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"      ✓ Saved visualization to {output_path}")

# ========== Step 6: Export detailed results ==========
print("\n[6/6] Exporting results...")

# JSON export
output_json = Path(__file__).parent / "reporting" / "transportability_statistical.json"
export_data = {
    "config": {
        "subsample_size": SUBSAMPLE_SIZE,
        "oracle_coverage": ORACLE_COVERAGE,
        "probe_size": PROBE_SIZE,
        "seed": SEED,
    },
    "calibrator": {
        "mode": mode,
        "rmse": float(cal_result.calibration_rmse),
    },
    "results": {
        policy: diag.to_dict() for policy, diag in results.items()
    }
}

with open(output_json, "w") as f:
    json.dump(export_data, f, indent=2)

print(f"      ✓ Saved JSON to {output_json}")

# LaTeX table export
output_tex = Path(__file__).parent / "reporting" / "transportability_table.tex"
with open(output_tex, "w") as f:
    f.write("% Transportability test results with statistical tests\n")
    f.write("\\begin{table}[htbp]\n")
    f.write("\\centering\n")
    f.write("\\caption{Transportability Test Results: Base Policy Calibrator → Target Policies}\n")
    f.write("\\label{tab:transportability}\n")
    f.write("\\begin{tabular}{lcccc}\n")
    f.write("\\toprule\n")
    f.write("Policy & Status & $\\hat{\\delta}$ & 95\\% CI & Coverage \\\\\n")
    f.write("\\midrule\n")

    for policy, diag in results.items():
        status_symbol = {
            "PASS": "\\checkmark",
            "WARN": "\\sim",
            "FAIL": "\\times"
        }[diag.status]

        ci_str = f"$[{diag.delta_ci[0]:+.3f}, {diag.delta_ci[1]:+.3f}]$"

        f.write(
            f"{policy.replace('_', ' ')} & "
            f"{status_symbol} & "
            f"${diag.delta_hat:+.3f}$ & "
            f"{ci_str} & "
            f"{diag.coverage:.1%} \\\\\n"
        )

    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"      ✓ Saved LaTeX table to {output_tex}")

# ========== Summary ==========
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

n_pass = sum(1 for d in results.values() if d.status == "PASS")
n_warn = sum(1 for d in results.values() if d.status == "WARN")
n_fail = sum(1 for d in results.values() if d.status == "FAIL")

print(f"\nResults: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL (out of {len(results)})")

# Unbiasedness test summary
print("\nUnbiasedness Test Summary (0 ∈ CI?):")
for policy, diag in results.items():
    zero_in_ci = diag.delta_ci[0] <= 0 <= diag.delta_ci[1]
    status_symbol = "✓" if zero_in_ci else "✗"
    print(f"  {policy}: δ̂={diag.delta_hat:+.4f}, CI=[{diag.delta_ci[0]:+.4f}, {diag.delta_ci[1]:+.4f}] {status_symbol}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

if n_pass == len(results):
    print("\n✓ Excellent! All target policies fall within baseline variation.")
    print("  The calibrator transports well - no significant deviation from base.")
elif n_fail == 0:
    print("\n⚠ Marginal transport. Some policies show small deviations from baseline.")
    print("  Consider monitoring or collecting more oracle data.")
else:
    print("\n✗ Poor transport detected.")
    print("  Failed policies show:")
    for policy, diag in results.items():
        if diag.status == "FAIL":
            print(f"    • {policy}: δ̂={diag.delta_hat:+.4f} (outside baseline)")
    print("\n  Recommended: Refit calibrator with pooled data or use two-stage calibration.")

print("\n" + "=" * 80)
