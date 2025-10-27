#!/usr/bin/env python3
"""Generate forest plot showing one estimator's CIs across policies vs oracle truth."""

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


def extract_estimator_data(
    results: List[Dict],
    estimator_name: str,
    sample_size: int,
    oracle_coverage: float,
    include_baseline: bool = False,
) -> Dict[str, List[Dict]]:
    """Extract data for a specific estimator in a specific scenario.

    Args:
        results: All experiment results
        estimator_name: Target estimator to extract
        sample_size: Sample size filter
        oracle_coverage: Oracle coverage filter
        include_baseline: If True, also extract naive-direct baseline

    Returns:
        Dictionary mapping estimator name -> list of results
    """

    matching_results = defaultdict(list)

    for r in results:
        spec = r["spec"]
        extra = spec.get("extra", {})

        # Filter by scenario
        if spec["sample_size"] != sample_size:
            continue
        if abs(spec["oracle_coverage"] - oracle_coverage) > 1e-6:
            continue

        # Check if this is the target estimator
        display_name = get_estimator_display_name(spec["estimator"], extra)
        if display_name == estimator_name:
            matching_results[estimator_name].append(r)

        # Also get baseline if requested
        if include_baseline and spec["estimator"] == "naive-direct":
            matching_results["naive-direct"].append(r)

    return dict(matching_results)


def extract_single_run_data(result: Dict) -> Dict[str, Dict]:
    """Extract policy data from a single experimental run.

    Args:
        result: Single experiment result dictionary

    Returns:
        Dictionary mapping policy -> {estimate, ci_lower, ci_upper, oracle_truth}
    """
    estimates = result.get("estimates", {})
    cis = result.get("confidence_intervals", {})
    oracles = result.get("oracle_truths", {})

    policy_data = {}

    # Extract data for policies with estimates
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

    # Add base policy oracle truth with computed CI
    if "base" in oracles:
        oracle_val = oracles["base"]
        n_oracle = result.get("oracle_slice_size", result.get("n_oracle", 250))

        # Compute binomial CI for oracle proportion
        # Using normal approximation: SE = sqrt(p(1-p)/n)
        se_oracle = np.sqrt(oracle_val * (1 - oracle_val) / n_oracle)
        ci_lower = oracle_val - 1.96 * se_oracle
        ci_upper = oracle_val + 1.96 * se_oracle

        policy_data["base"] = {
            "estimate": oracle_val,  # Use oracle truth as the estimate
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "oracle_truth": oracle_val,
            "n_oracle": n_oracle,  # Store for reference
        }

    return policy_data


def create_forest_plot_by_policy(
    policy_data: Dict[str, Dict],
    estimator_name: str,
    sample_size: int,
    oracle_coverage: float,
    seed: Optional[int] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Create forest plot showing one estimator's estimates across policies.

    Args:
        policy_data: Data from extract_single_run_data
        estimator_name: Name of estimator for title
        sample_size: Sample size for title
        oracle_coverage: Oracle coverage for title
        seed: Seed number for title (if specified)
        output_path: Where to save the figure
    """

    if not policy_data:
        print("No data to plot")
        return None

    # Sort policies alphabetically for consistent display
    policies = sorted(policy_data.keys())

    # Extract plot data
    plot_data = []
    for policy in policies:
        data = policy_data[policy]
        plot_data.append({
            "policy": policy,
            "estimate": data["estimate"],
            "ci_lower": data["ci_lower"],
            "ci_upper": data["ci_upper"],
            "oracle": data["oracle_truth"],
        })

    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_data) * 0.5)))

    # Extract data for plotting
    policies = [d["policy"] for d in plot_data]
    oracles = [d["oracle"] for d in plot_data]

    # Y positions (reverse so first policy is at top)
    y_pos = np.arange(len(policies))[::-1]

    # Plot oracle truth for all policies
    ax.scatter(oracles, y_pos, color='red', marker='d', s=100,
              label='Oracle truth', zorder=3, alpha=0.8, edgecolors='darkred', linewidths=1.5)

    # Plot estimates with error bars for all policies
    estimates_to_plot = []
    yerr_lower_to_plot = []
    yerr_upper_to_plot = []
    y_pos_to_plot = []

    for i, d in enumerate(plot_data):
        if d["estimate"] is not None:
            estimates_to_plot.append(d["estimate"])

            if d["ci_lower"] is not None and d["ci_upper"] is not None:
                yerr_lower_to_plot.append(d["estimate"] - d["ci_lower"])
                yerr_upper_to_plot.append(d["ci_upper"] - d["estimate"])
            else:
                yerr_lower_to_plot.append(0)
                yerr_upper_to_plot.append(0)

            y_pos_to_plot.append(y_pos[i])

    if estimates_to_plot:
        ax.errorbar(estimates_to_plot, y_pos_to_plot,
                    xerr=[yerr_lower_to_plot, yerr_upper_to_plot],
                    fmt='o', markersize=8, capsize=5, capthick=2,
                    color='steelblue', ecolor='steelblue',
                    label='Estimate ± 95% CI', zorder=2)

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(policies, fontsize=11)
    ax.set_xlabel('Oracle Score', fontsize=12)
    ax.set_ylabel('Policy', fontsize=12)

    # Title
    if seed is not None:
        title = f'Forest Plot: {estimator_name}\n(n={sample_size}, oracle={int(oracle_coverage*100)}%, seed={seed})'
    else:
        title = f'Forest Plot: {estimator_name}\n(n={sample_size}, oracle={int(oracle_coverage*100)}%)'
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
    """Generate forest plot for one estimator across policies."""

    print("="*70)
    print("FOREST PLOT BY POLICY")
    print("="*70)

    # Configuration - change these to select different scenarios
    ESTIMATOR = "direct+cov"  # Change this to any estimator
    SAMPLE_SIZE = 1000
    ORACLE_COVERAGE = 0.25  # 25%
    SEED_INDEX = 0  # Which run to plot (0 = first seed)

    print(f"\nConfiguration:")
    print(f"  Estimator: {ESTIMATOR}")
    print(f"  Sample size: {SAMPLE_SIZE}")
    print(f"  Oracle coverage: {int(ORACLE_COVERAGE*100)}%")
    print(f"  Seed index: {SEED_INDEX}")

    # Load results
    print("\nLoading results...")
    results = load_results()
    print(f"Loaded {len(results)} successful experiments")

    # Extract data for this estimator
    print(f"\nExtracting data for {ESTIMATOR}...")
    results_by_estimator = extract_estimator_data(results, ESTIMATOR, SAMPLE_SIZE, ORACLE_COVERAGE)

    if ESTIMATOR not in results_by_estimator:
        print(f"No data found for {ESTIMATOR} in this scenario!")
        return

    estimator_results = results_by_estimator[ESTIMATOR]
    print(f"Found {len(estimator_results)} runs")

    if SEED_INDEX >= len(estimator_results):
        print(f"Seed index {SEED_INDEX} out of range (only {len(estimator_results)} runs available)")
        return

    # Extract single run data
    print(f"\nExtracting data from run {SEED_INDEX}...")
    selected_run = estimator_results[SEED_INDEX]
    seed = selected_run.get("seed")
    policy_data = extract_single_run_data(selected_run)
    print(f"Seed: {seed}")
    print(f"Found {len(policy_data)} policies")

    # Create forest plot
    print("\nCreating forest plot...")
    output_path = Path(f"results/analysis/forest_by_policy_{ESTIMATOR.replace('+', '_')}_n{SAMPLE_SIZE}_oracle{int(ORACLE_COVERAGE*100)}_seed{seed}.png")

    fig = create_forest_plot_by_policy(
        policy_data,
        ESTIMATOR,
        SAMPLE_SIZE,
        ORACLE_COVERAGE,
        seed=seed,
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
