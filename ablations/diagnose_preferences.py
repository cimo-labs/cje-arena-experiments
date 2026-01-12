#!/usr/bin/env python3
"""
Diagnostics and analysis for preferences ablation.

Performs transport audits, computes Youden's J, and generates
comparison tables for CJE vs Rogan-Gladen.

Usage:
    python diagnose_preferences.py
    python diagnose_preferences.py --report
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from scipy import stats

from config_preferences import (
    RESULTS_DIR_PREFERENCES,
    RESULTS_PATH_PREFERENCES,
    BINARY_RESULTS_PATH,
    BINARY_SUMMARY_PATH,
    RESPONSES_DIR_PREFERENCES,
    TARGET_POLICIES_PREFERENCES,
    ALL_POLICIES_PREFERENCES,
)


def load_continuous_results() -> List[Dict[str, Any]]:
    """Load continuous CJE experiment results."""
    results = []
    if RESULTS_PATH_PREFERENCES.exists():
        with open(RESULTS_PATH_PREFERENCES, 'r') as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return [r for r in results if r.get("success")]


def load_binary_results() -> List[Dict[str, Any]]:
    """Load binary baseline experiment results."""
    results = []
    if BINARY_RESULTS_PATH.exists():
        with open(BINARY_RESULTS_PATH, 'r') as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return [r for r in results if r.get("success")]


def compute_transport_audit_continuous(
    results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Compute transport audit for continuous CJE methods.

    The transport audit checks if the calibration learned on base policy
    transfers to target policies. Pass condition: mean residual E[Y - f(S)] = 0.

    We operationalize this by checking if the mean estimate across seeds
    matches the oracle truth (within statistical tolerance).
    """
    audit_results = {}

    # Group by policy
    for policy in TARGET_POLICIES_PREFERENCES:
        policy_estimates = {"naive-direct": [], "direct": []}
        oracle_truth = None

        for r in results:
            estimator = r.get("spec", {}).get("estimator", "")
            estimates = r.get("estimates", {})
            truths = r.get("oracle_truths", {})

            if policy in estimates and estimator in policy_estimates:
                policy_estimates[estimator].append(estimates[policy])

            if policy in truths and oracle_truth is None:
                oracle_truth = truths[policy]

        if oracle_truth is None:
            continue

        audit_results[policy] = {"oracle_truth": oracle_truth}

        for estimator, estimates in policy_estimates.items():
            if not estimates:
                continue

            mean_est = np.mean(estimates)
            std_est = np.std(estimates, ddof=1)
            se_est = std_est / np.sqrt(len(estimates))

            # Bias
            bias = mean_est - oracle_truth
            bias_se = se_est

            # Transport pass/fail: |bias| < 2*SE
            passes = abs(bias) < 2 * bias_se if bias_se > 0 else True

            audit_results[policy][estimator] = {
                "mean_estimate": float(mean_est),
                "std_estimate": float(std_est),
                "bias": float(bias),
                "bias_se": float(bias_se),
                "transport_passes": bool(passes),
                "n_experiments": len(estimates),
            }

    return audit_results


def compute_confusion_matrix_audit(
    results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Compute confusion matrix audit for binary methods.

    The binary transport audit checks if the confusion matrix (Se, Sp)
    is stable across policies. If Se or Sp varies significantly,
    the Rogan-Gladen adjustment will fail.
    """
    audit_results = {}

    for policy in ALL_POLICIES_PREFERENCES:
        sensitivities = []
        specificities = []
        youden_js = []

        for r in results:
            cm = r.get("confusion_matrix", {}).get(policy, {})
            if "sensitivity" in cm:
                sensitivities.append(cm["sensitivity"])
                specificities.append(cm["specificity"])
                youden_js.append(cm["youden_j"])

        if not youden_js:
            continue

        mean_se = np.mean(sensitivities)
        mean_sp = np.mean(specificities)
        mean_j = np.mean(youden_js)

        # Stability check: CV < 0.2 is stable
        cv_se = np.std(sensitivities) / mean_se if mean_se > 0 else 1
        cv_sp = np.std(specificities) / mean_sp if mean_sp > 0 else 1

        # Transport passes if CV is low and J > 0.3
        stable = cv_se < 0.2 and cv_sp < 0.2
        j_valid = mean_j > 0.3

        audit_results[policy] = {
            "mean_sensitivity": float(mean_se),
            "mean_specificity": float(mean_sp),
            "mean_youden_j": float(mean_j),
            "cv_sensitivity": float(cv_se),
            "cv_specificity": float(cv_sp),
            "confusion_stable": bool(stable),
            "j_valid": bool(j_valid),
            "transport_passes": bool(stable and j_valid),
            "n_experiments": len(youden_js),
        }

    return audit_results


def compute_rmse_comparison(
    continuous_results: List[Dict[str, Any]],
    binary_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Compare RMSE across methods."""
    comparison = {}

    # Continuous RMSE by estimator
    for estimator in ["naive-direct", "direct"]:
        rmses = [
            r.get("rmse_vs_oracle")
            for r in continuous_results
            if r.get("spec", {}).get("estimator") == estimator
            and r.get("rmse_vs_oracle") is not None
        ]
        if rmses:
            comparison[estimator] = {
                "mean_rmse": float(np.mean(rmses)),
                "std_rmse": float(np.std(rmses)),
                "n": len(rmses),
            }

    # Binary RMSE
    naive_rmses = [r.get("rmse_naive") for r in binary_results if r.get("rmse_naive") is not None]
    adjusted_rmses = [r.get("rmse_adjusted") for r in binary_results if r.get("rmse_adjusted") is not None]

    if naive_rmses:
        comparison["binary-naive"] = {
            "mean_rmse": float(np.mean(naive_rmses)),
            "std_rmse": float(np.std(naive_rmses)),
            "n": len(naive_rmses),
        }

    if adjusted_rmses:
        comparison["binary-adjusted"] = {
            "mean_rmse": float(np.mean(adjusted_rmses)),
            "std_rmse": float(np.std(adjusted_rmses)),
            "n": len(adjusted_rmses),
        }

    return comparison


def compute_ranking_accuracy(
    continuous_results: List[Dict[str, Any]],
    binary_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Compute ranking accuracy (Kendall τ) for each method."""
    from scipy.stats import kendalltau

    ranking_results = {}

    # Get oracle ranking (by mean across experiments)
    oracle_means = {}
    for r in continuous_results:
        for policy, value in r.get("oracle_truths", {}).items():
            if policy not in oracle_means:
                oracle_means[policy] = []
            oracle_means[policy].append(value)

    oracle_ranking = sorted(
        [(p, np.mean(v)) for p, v in oracle_means.items()],
        key=lambda x: -x[1]
    )
    oracle_order = [p for p, _ in oracle_ranking]

    if len(oracle_order) < 2:
        return ranking_results

    # Continuous methods
    for estimator in ["naive-direct", "direct"]:
        taus = []
        for r in continuous_results:
            if r.get("spec", {}).get("estimator") != estimator:
                continue

            estimates = r.get("estimates", {})
            if len(estimates) < 2:
                continue

            # Compute ranking
            est_ranking = sorted(
                [(p, v) for p, v in estimates.items() if p in oracle_order],
                key=lambda x: -x[1]
            )
            est_order = [p for p, _ in est_ranking]

            # Kendall τ
            oracle_ranks = [oracle_order.index(p) for p in est_order if p in oracle_order]
            est_ranks = list(range(len(est_order)))

            if len(oracle_ranks) >= 2:
                tau, _ = kendalltau(oracle_ranks, est_ranks)
                if not np.isnan(tau):
                    taus.append(tau)

        if taus:
            ranking_results[estimator] = {
                "mean_tau": float(np.mean(taus)),
                "std_tau": float(np.std(taus)),
                "n": len(taus),
            }

    # Binary methods
    for method, est_key in [("binary-naive", "estimates_naive"), ("binary-adjusted", "estimates_adjusted")]:
        taus = []
        for r in binary_results:
            estimates = r.get(est_key, {})
            if len(estimates) < 2:
                continue

            est_ranking = sorted(
                [(p, v) for p, v in estimates.items() if p in oracle_order],
                key=lambda x: -x[1]
            )
            est_order = [p for p, _ in est_ranking]

            oracle_ranks = [oracle_order.index(p) for p in est_order if p in oracle_order]
            est_ranks = list(range(len(est_order)))

            if len(oracle_ranks) >= 2:
                tau, _ = kendalltau(oracle_ranks, est_ranks)
                if not np.isnan(tau):
                    taus.append(tau)

        if taus:
            ranking_results[method] = {
                "mean_tau": float(np.mean(taus)),
                "std_tau": float(np.std(taus)),
                "n": len(taus),
            }

    return ranking_results


def compute_bias_corrected_rmse(
    results: List[Dict[str, Any]],
    transport_audit: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Apply transport-based bias correction to CJE estimates.

    Uses the bias from the transport audit to correct each policy's estimate:
        corrected_estimate = raw_estimate - bias

    This should reduce RMSE when transport fails (e.g., contrarian policy).
    """
    corrected_rmses = []

    for r in results:
        if r.get("spec", {}).get("estimator") != "direct":
            continue

        estimates = r.get("estimates", {})
        truths = r.get("oracle_truths", {})

        corrected_squared_errors = []
        for policy in estimates:
            if policy not in transport_audit:
                continue

            # Get bias from transport audit
            bias = transport_audit[policy].get("direct", {}).get("bias", 0)

            # Correct the estimate
            raw_estimate = estimates[policy]
            corrected_estimate = raw_estimate - bias

            # Compute squared error vs oracle
            oracle = truths.get(policy, 0.505)
            se = (corrected_estimate - oracle) ** 2
            corrected_squared_errors.append(se)

        if corrected_squared_errors:
            rmse = np.sqrt(np.mean(corrected_squared_errors))
            corrected_rmses.append(rmse)

    if not corrected_rmses:
        return {}

    return {
        "direct-corrected": {
            "mean_rmse": float(np.mean(corrected_rmses)),
            "std_rmse": float(np.std(corrected_rmses)),
            "n": len(corrected_rmses),
        }
    }


def generate_report(
    continuous_results: List[Dict[str, Any]],
    binary_results: List[Dict[str, Any]],
) -> str:
    """Generate markdown report."""
    lines = []
    lines.append("# Preferences Ablation: CJE vs Binary Calibration Results")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Continuous experiments: {len(continuous_results)}")
    lines.append(f"- Binary experiments: {len(binary_results)}")
    lines.append("")

    # RMSE comparison
    rmse_comparison = compute_rmse_comparison(continuous_results, binary_results)
    lines.append("## RMSE Comparison")
    lines.append("")
    lines.append("| Method | Mean RMSE | Std RMSE | N |")
    lines.append("|--------|-----------|----------|---|")
    for method in ["naive-direct", "direct", "binary-naive", "binary-adjusted"]:
        if method in rmse_comparison:
            stats = rmse_comparison[method]
            lines.append(f"| {method} | {stats['mean_rmse']:.4f} | {stats['std_rmse']:.4f} | {stats['n']} |")
    lines.append("")

    # Transport audits
    continuous_audit = compute_transport_audit_continuous(continuous_results)
    lines.append("## Continuous Transport Audit")
    lines.append("")
    lines.append("| Policy | Oracle | naive-direct Bias | direct Bias | direct Passes |")
    lines.append("|--------|--------|-------------------|-------------|---------------|")
    for policy, audit in continuous_audit.items():
        oracle = audit.get("oracle_truth", "N/A")
        naive = audit.get("naive-direct", {})
        direct = audit.get("direct", {})
        lines.append(
            f"| {policy} | {oracle:.3f} | "
            f"{naive.get('bias', 'N/A'):.3f} | "
            f"{direct.get('bias', 'N/A'):.3f} | "
            f"{'PASS' if direct.get('transport_passes', False) else 'FAIL'} |"
        )
    lines.append("")

    # Bias-corrected RMSE
    bias_corrected = compute_bias_corrected_rmse(continuous_results, continuous_audit)
    if bias_corrected and "direct" in rmse_comparison:
        lines.append("## Bias-Corrected RMSE (Transport Residual Adjustment)")
        lines.append("")
        lines.append("| Method | Raw RMSE | Bias-Corrected RMSE | Improvement |")
        lines.append("|--------|----------|---------------------|-------------|")
        raw_rmse = rmse_comparison["direct"]["mean_rmse"]
        corrected_rmse = bias_corrected["direct-corrected"]["mean_rmse"]
        improvement = (raw_rmse - corrected_rmse) / raw_rmse * 100
        lines.append(
            f"| CJE (direct) | {raw_rmse:.4f} | {corrected_rmse:.4f} | "
            f"{improvement:+.1f}% |"
        )
        lines.append("")
        lines.append("*Correction formula: `corrected = estimate - transport_bias`*")
        lines.append("")

    # Binary confusion matrix audit
    binary_audit = compute_confusion_matrix_audit(binary_results)
    if binary_audit:
        lines.append("## Binary Confusion Matrix Audit")
        lines.append("")
        lines.append("| Policy | Sensitivity | Specificity | Youden's J | Transport |")
        lines.append("|--------|-------------|-------------|------------|-----------|")
        for policy, audit in binary_audit.items():
            lines.append(
                f"| {policy} | {audit['mean_sensitivity']:.3f} | "
                f"{audit['mean_specificity']:.3f} | "
                f"{audit['mean_youden_j']:.3f} | "
                f"{'PASS' if audit['transport_passes'] else 'FAIL'} |"
            )
        lines.append("")

        # Youden's J comparison to MBPP
        lines.append("### Youden's J Comparison")
        lines.append("")
        lines.append("| Domain | Mean Youden's J | Binary Works? |")
        lines.append("|--------|-----------------|---------------|")
        lines.append("| MBPP (code) | 0.054 | No |")
        mean_j = np.mean([a["mean_youden_j"] for a in binary_audit.values()])
        works = "Yes" if mean_j > 0.3 else "No"
        lines.append(f"| Arena (prefs) | {mean_j:.3f} | {works} |")
        lines.append("")

    # Ranking accuracy
    ranking = compute_ranking_accuracy(continuous_results, binary_results)
    if ranking:
        lines.append("## Ranking Accuracy (Kendall τ)")
        lines.append("")
        lines.append("| Method | Mean τ | Std τ | N |")
        lines.append("|--------|--------|-------|---|")
        for method in ["naive-direct", "direct", "binary-naive", "binary-adjusted"]:
            if method in ranking:
                stats = ranking[method]
                lines.append(f"| {method} | {stats['mean_tau']:.3f} | {stats['std_tau']:.3f} | {stats['n']} |")
        lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")

    # Determine winner
    if rmse_comparison:
        best_method = min(rmse_comparison.items(), key=lambda x: x[1]["mean_rmse"])[0]
        lines.append(f"1. **Best RMSE**: {best_method}")

        if "binary-adjusted" in rmse_comparison and "direct" in rmse_comparison:
            binary_rmse = rmse_comparison["binary-adjusted"]["mean_rmse"]
            cje_rmse = rmse_comparison["direct"]["mean_rmse"]
            if binary_rmse < cje_rmse:
                lines.append(f"2. **Binary calibration works on preferences** (RMSE: {binary_rmse:.4f} vs {cje_rmse:.4f})")
            else:
                pct_improvement = (binary_rmse - cje_rmse) / binary_rmse * 100
                lines.append(f"2. **CJE beats binary** by {pct_improvement:.1f}%")

    if binary_audit:
        mean_j = np.mean([a["mean_youden_j"] for a in binary_audit.values()])
        lines.append(f"3. **Youden's J = {mean_j:.3f}** (vs 0.054 on MBPP)")
        if mean_j > 0.3:
            lines.append("   - Binary calibration is valid on preferences (J > 0.3)")
        else:
            lines.append("   - Binary calibration is unstable (J < 0.3)")

    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Diagnostics for preferences ablation")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate and save markdown report",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON diagnostics",
    )
    args = parser.parse_args()

    # Load results
    continuous_results = load_continuous_results()
    binary_results = load_binary_results()

    print(f"Loaded {len(continuous_results)} continuous experiments")
    print(f"Loaded {len(binary_results)} binary experiments")

    if not continuous_results and not binary_results:
        print("No results found. Run run_preferences.py first.")
        return 1

    if args.json:
        diagnostics = {
            "continuous_transport_audit": compute_transport_audit_continuous(continuous_results),
            "binary_confusion_audit": compute_confusion_matrix_audit(binary_results),
            "rmse_comparison": compute_rmse_comparison(continuous_results, binary_results),
            "ranking_accuracy": compute_ranking_accuracy(continuous_results, binary_results),
        }
        print(json.dumps(diagnostics, indent=2))
        return 0

    # Generate report
    report = generate_report(continuous_results, binary_results)
    print(report)

    if args.report:
        report_path = RESULTS_DIR_PREFERENCES / "REPORT.md"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
