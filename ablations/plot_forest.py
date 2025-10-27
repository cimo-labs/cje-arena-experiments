#!/usr/bin/env python3
"""Generate forest plot showing estimator CIs vs oracle truth for a specific scenario."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict


def load_results(path: str = "results/all_experiments.jsonl") -> List[Dict]:
    """Load experiment results from the unified ablation output."""
    results = []
    with open(path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get("success"):
                    results.append(data)
            except:
                pass
    return results


def get_estimator_display_name(estimator: str, extra: Dict[str, Any]) -> str:
    """Create display name for estimator."""

    # Check for covariates
    use_cov = extra.get("use_covariates", False)
    cov_suffix = "+cov" if use_cov else ""

    # Check for calibration
    use_cal = extra.get("use_weight_calibration", False)

    # Map internal names to display names
    name_map = {
        "direct": f"direct{cov_suffix}",
        "raw-ips": f"SNIPS{cov_suffix}",
        "calibrated-ips": f"calibrated-ips{cov_suffix}",
        "dr-cpo": f"dr-cpo{cov_suffix}" if not use_cal else f"calibrated-dr-cpo{cov_suffix}",
        "tr-cpo-e": f"tr-cpo-e{cov_suffix}",
        "stacked-dr": f"stacked-dr{cov_suffix}",
        "naive-direct": "naive-direct",
    }

    return name_map.get(estimator, f"{estimator}{cov_suffix}")


def extract_scenario_data(
    results: List[Dict],
    sample_size: int,
    oracle_coverage: float,
    policies: Optional[List[str]] = None,
) -> Dict[str, List[Dict]]:
    """Extract data for a specific scenario.

    Returns:
        Dictionary mapping estimator name to list of results (across seeds)
    """

    by_estimator = defaultdict(list)

    for r in results:
        spec = r["spec"]
        extra = spec.get("extra", {})

        # Filter by scenario
        if spec["sample_size"] != sample_size:
            continue
        if abs(spec["oracle_coverage"] - oracle_coverage) > 1e-6:
            continue

        estimator = spec["estimator"]
        display_name = get_estimator_display_name(estimator, extra)

        # Extract per-policy data
        estimates = r.get("estimates", {})
        cis = r.get("confidence_intervals", {})
        oracle = r.get("oracle_truths", {})

        # Filter to requested policies
        if policies:
            estimates = {k: v for k, v in estimates.items() if k in policies}
            cis = {k: v for k, v in cis.items() if k in policies}
            oracle = {k: v for k, v in oracle.items() if k in policies}

        # Exclude unhelpful policy
        estimates = {k: v for k, v in estimates.items() if k != "unhelpful"}
        cis = {k: v for k, v in cis.items() if k != "unhelpful"}
        oracle = {k: v for k, v in oracle.items() if k != "unhelpful"}

        by_estimator[display_name].append({
            "estimates": estimates,
            "confidence_intervals": cis,
            "oracle_truths": oracle,
        })

    return dict(by_estimator)


def aggregate_across_seeds(results_by_estimator: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Aggregate results across seeds for each estimator.

    Returns:
        Dictionary with mean estimates, CIs, and oracle truths per estimator per policy
    """

    aggregated = {}

    for estimator, runs in results_by_estimator.items():
        if not runs:
            continue

        # Get all policies
        all_policies = set()
        for run in runs:
            all_policies.update(run["estimates"].keys())

        # Aggregate each policy
        policy_data = {}
        for policy in all_policies:
            estimates = []
            ci_lowers = []
            ci_uppers = []
            oracles = []

            for run in runs:
                if policy in run["estimates"]:
                    est = run["estimates"][policy]
                    if np.isfinite(est):
                        estimates.append(est)

                        if policy in run["confidence_intervals"]:
                            ci_lower, ci_upper = run["confidence_intervals"][policy]
                            ci_lowers.append(ci_lower)
                            ci_uppers.append(ci_upper)

                        if policy in run["oracle_truths"]:
                            oracles.append(run["oracle_truths"][policy])

            if estimates:
                policy_data[policy] = {
                    "mean_estimate": np.mean(estimates),
                    "mean_ci_lower": np.mean(ci_lowers) if ci_lowers else None,
                    "mean_ci_upper": np.mean(ci_uppers) if ci_uppers else None,
                    "oracle_truth": np.mean(oracles) if oracles else None,
                    "n_seeds": len(estimates),
                }

        aggregated[estimator] = policy_data

    return aggregated


def create_forest_plot(
    aggregated_data: Dict[str, Dict],
    sample_size: int,
    oracle_coverage: float,
    policy: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create forest plot showing estimates and CIs vs oracle truth.

    Args:
        aggregated_data: Data from aggregate_across_seeds
        sample_size: Sample size for title
        oracle_coverage: Oracle coverage for title
        policy: Specific policy to plot (if None, averages across policies)
        output_path: Where to save the figure
    """

    # Determine estimator order (by methodology)
    estimator_order = [
        "naive-direct", "direct", "direct+cov",
        "SNIPS", "SNIPS+cov", "calibrated-ips", "calibrated-ips+cov",
        "dr-cpo", "dr-cpo+cov", "calibrated-dr-cpo", "calibrated-dr-cpo+cov",
        "stacked-dr", "stacked-dr+cov",
        "tr-cpo-e", "tr-cpo-e+cov",
    ]

    # Filter to available estimators
    available_estimators = [e for e in estimator_order if e in aggregated_data]

    if not available_estimators:
        print("No estimators found in data")
        return None

    # Extract plot data
    plot_data = []

    for estimator in available_estimators:
        policy_data = aggregated_data[estimator]

        if policy:
            # Single policy
            if policy not in policy_data:
                continue
            data = policy_data[policy]
            plot_data.append({
                "estimator": estimator,
                "estimate": data["mean_estimate"],
                "ci_lower": data["mean_ci_lower"],
                "ci_upper": data["mean_ci_upper"],
                "oracle": data["oracle_truth"],
            })
        else:
            # Average across policies
            estimates = []
            ci_lowers = []
            ci_uppers = []
            oracles = []

            for pol, data in policy_data.items():
                estimates.append(data["mean_estimate"])
                if data["mean_ci_lower"] is not None:
                    ci_lowers.append(data["mean_ci_lower"])
                if data["mean_ci_upper"] is not None:
                    ci_uppers.append(data["mean_ci_upper"])
                if data["oracle_truth"] is not None:
                    oracles.append(data["oracle_truth"])

            if estimates:
                plot_data.append({
                    "estimator": estimator,
                    "estimate": np.mean(estimates),
                    "ci_lower": np.mean(ci_lowers) if ci_lowers else None,
                    "ci_upper": np.mean(ci_uppers) if ci_uppers else None,
                    "oracle": np.mean(oracles) if oracles else None,
                })

    if not plot_data:
        print("No data to plot")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_data) * 0.4)))

    # Extract data for plotting
    estimators = [d["estimator"] for d in plot_data]
    estimates = [d["estimate"] for d in plot_data]
    ci_lowers = [d["ci_lower"] for d in plot_data]
    ci_uppers = [d["ci_upper"] for d in plot_data]
    oracles = [d["oracle"] for d in plot_data]

    # Calculate error bars
    yerr_lower = [est - ci_l if ci_l is not None else 0 for est, ci_l in zip(estimates, ci_lowers)]
    yerr_upper = [ci_u - est if ci_u is not None else 0 for est, ci_u in zip(estimates, ci_uppers)]

    # Y positions (reverse so first estimator is at top)
    y_pos = np.arange(len(estimators))[::-1]

    # Plot oracle truth as vertical reference line (if consistent)
    oracle_vals = [o for o in oracles if o is not None]
    if oracle_vals:
        oracle_mean = np.mean(oracle_vals)
        oracle_std = np.std(oracle_vals) if len(oracle_vals) > 1 else 0

        if oracle_std < 0.01:  # Consistent oracle truth
            ax.axvline(oracle_mean, color='red', linestyle='--', linewidth=2,
                      label=f'Oracle truth: {oracle_mean:.3f}', zorder=1)
        else:
            # Variable oracle truth - plot as points
            ax.scatter(oracles, y_pos, color='red', marker='d', s=80,
                      label='Oracle truth', zorder=3, alpha=0.7)

    # Plot estimates with error bars
    ax.errorbar(estimates, y_pos,
                xerr=[yerr_lower, yerr_upper],
                fmt='o', markersize=8, capsize=5, capthick=2,
                color='steelblue', ecolor='steelblue',
                label='Estimate ± 95% CI', zorder=2)

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(estimators, fontsize=10)
    ax.set_xlabel('Estimated Value', fontsize=12)
    ax.set_ylabel('Estimator', fontsize=12)

    # Title
    if policy:
        title = f'Forest Plot: {policy}\n(n={sample_size}, oracle={int(oracle_coverage*100)}%)'
    else:
        title = f'Forest Plot: Average Across Policies\n(n={sample_size}, oracle={int(oracle_coverage*100)}%)'

    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    # Legend
    ax.legend(loc='best', fontsize=10)

    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle=':')

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved forest plot to {output_path}")

    return fig


def main():
    """Generate forest plot for a specific scenario."""

    print("="*70)
    print("FOREST PLOT GENERATION")
    print("="*70)

    # Configuration - change these to select different scenarios
    SAMPLE_SIZE = 1000
    ORACLE_COVERAGE = 0.25  # 25%
    POLICY = None  # None = average across policies, or specify like "gpt-4o"

    print(f"\nConfiguration:")
    print(f"  Sample size: {SAMPLE_SIZE}")
    print(f"  Oracle coverage: {int(ORACLE_COVERAGE*100)}%")
    print(f"  Policy: {POLICY if POLICY else 'Average across policies'}")

    # Load results
    print("\nLoading results...")
    results = load_results()
    print(f"Loaded {len(results)} successful experiments")

    # Extract scenario data
    print(f"\nExtracting data for scenario...")
    scenario_data = extract_scenario_data(results, SAMPLE_SIZE, ORACLE_COVERAGE)
    print(f"Found {len(scenario_data)} estimators with data")

    if not scenario_data:
        print("No data found for this scenario!")
        return

    # Aggregate across seeds
    print("\nAggregating across seeds...")
    aggregated = aggregate_across_seeds(scenario_data)

    # Create forest plot
    print("\nCreating forest plot...")
    output_path = Path(f"results/analysis/forest_plot_n{SAMPLE_SIZE}_oracle{int(ORACLE_COVERAGE*100)}.png")

    fig = create_forest_plot(
        aggregated,
        SAMPLE_SIZE,
        ORACLE_COVERAGE,
        policy=POLICY,
        output_path=output_path,
    )

    if fig:
        print("\n" + "="*70)
        print("FOREST PLOT GENERATION COMPLETE")
        print("="*70)
    else:
        print("\nFailed to generate forest plot")


if __name__ == "__main__":
    main()
