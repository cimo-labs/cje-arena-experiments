#!/usr/bin/env python3
"""Generate MDE plot for direct+cov estimator."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional


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


def analyze_interaction(
    results: List[Dict],
    estimator: str = "direct",
    use_covariates: bool = True,
) -> Dict[str, Any]:
    """
    Analyze interaction effects between oracle coverage and sample size.

    Args:
        results: List of experiment results
        estimator: Which estimator to analyze
        use_covariates: Whether to filter for covariates

    Returns:
        Analysis dictionary with grids and statistics
    """

    # Filter to relevant results
    filtered = []
    for r in results:
        spec = r["spec"]
        extra = spec.get("extra", {})

        # Check estimator match
        if spec["estimator"] != estimator:
            continue

        # Check covariate filter
        if extra.get("use_covariates", False) != use_covariates:
            continue

        filtered.append(r)

    print(f"Analyzing {len(filtered)} results for {estimator} (covariates={use_covariates})")

    # Build grids
    rmse_grid: Dict[tuple, list] = {}
    se_grid: Dict[tuple, list] = {}

    for r in filtered:
        oracle = r["spec"]["oracle_coverage"]
        n_samples = r["spec"]["sample_size"]
        key = (oracle, n_samples)

        if key not in rmse_grid:
            rmse_grid[key] = []
            se_grid[key] = []

        rmse_grid[key].append(r.get("rmse_vs_oracle", np.nan))

        # Get standard errors
        if "standard_errors" in r and r["standard_errors"]:
            se_dict = r["standard_errors"]
            # Exclude unhelpful policy
            se_no_unhelpful = {k: v for k, v in se_dict.items() if k != "unhelpful"}
            avg_se = np.nanmean(list(se_no_unhelpful.values()))
            se_grid[key].append(avg_se)

    # Average across seeds/runs
    mean_rmse = {k: np.nanmean(v) for k, v in rmse_grid.items()}
    mean_se = {k: np.nanmean(v) if v else np.nan for k, v in se_grid.items()}

    # Extract unique values for axes
    oracle_values = sorted(set(k[0] for k in mean_rmse.keys()))
    sample_values = sorted(set(k[1] for k in mean_rmse.keys()))

    # Create matrices
    rmse_matrix = np.full((len(oracle_values), len(sample_values)), np.nan)
    se_matrix = np.full((len(oracle_values), len(sample_values)), np.nan)

    for i, oracle in enumerate(oracle_values):
        for j, n_samples in enumerate(sample_values):
            if (oracle, n_samples) in mean_rmse:
                rmse_matrix[i, j] = mean_rmse[(oracle, n_samples)]
                se_matrix[i, j] = mean_se.get((oracle, n_samples), np.nan)

    # Compute MDE (Minimum Detectable Effect)
    # For 80% power at 95% confidence
    z_alpha = 1.96  # 95% CI
    z_power = 0.84  # 80% power
    k = z_alpha + z_power  # ≈ 2.80

    # MDE for two-policy comparison
    mde_two = k * np.sqrt(2.0) * se_matrix

    # Find sweet spots
    sweet_spots = []
    target_mdes = [0.01, 0.02, 0.05]  # 1%, 2%, 5% effect sizes

    for i, oracle in enumerate(oracle_values):
        for j, n_samples in enumerate(sample_values):
            if np.isfinite(se_matrix[i, j]):
                n_oracle = oracle * n_samples
                mde = mde_two[i, j]

                achievable = [t for t in target_mdes if mde <= t]

                if achievable:
                    cost_per_percent = n_oracle / (min(achievable) * 100)

                    sweet_spots.append({
                        "oracle_coverage": oracle,
                        "sample_size": n_samples,
                        "n_oracle": n_oracle,
                        "rmse": rmse_matrix[i, j],
                        "mde": mde,
                        "achievable_mde": min(achievable),
                        "cost_efficiency": cost_per_percent,
                    })

    sweet_spots.sort(key=lambda x: x["cost_efficiency"])

    return {
        "oracle_values": oracle_values,
        "sample_values": sample_values,
        "rmse_matrix": rmse_matrix,
        "se_matrix": se_matrix,
        "mde_matrix": mde_two,
        "sweet_spots": sweet_spots[:5],
    }


def create_mde_plot(
    analysis: Dict[str, Any],
    title: str,
    output_path: Optional[Path] = None
) -> Optional[plt.Figure]:
    """
    Create MDE plot for direct+cov.

    Args:
        analysis: Results from analyze_interaction
        title: Plot title
        output_path: Where to save the figure

    Returns:
        matplotlib Figure object
    """

    if not analysis["oracle_values"] or not analysis["sample_values"]:
        print("No data to plot")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Common formatting
    oracle_labels = [f"{int(100*y)}%" for y in analysis["oracle_values"]]
    sample_labels = [str(x) for x in analysis["sample_values"]]

    # Panel A: RMSE heatmap
    ax = axes[0]
    rmse = analysis["rmse_matrix"]
    mask = ~np.isfinite(rmse)

    rmse_flipped = np.flipud(rmse)
    mask_flipped = np.flipud(mask)
    oracle_labels_flipped = oracle_labels[::-1]

    sns.heatmap(
        rmse_flipped,
        mask=mask_flipped,
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=np.nanquantile(rmse, 0.95) if np.any(np.isfinite(rmse)) else 0.05,
        xticklabels=sample_labels,
        yticklabels=oracle_labels_flipped,
        cbar_kws={"label": "RMSE"},
        ax=ax,
    )
    ax.set_xlabel("Sample Size", fontsize=14, fontweight="bold")
    ax.set_ylabel("Oracle Coverage", fontsize=14, fontweight="bold")
    ax.set_title("A. RMSE", fontsize=15, fontweight="bold", pad=15)

    # Panel B: Standard Error heatmap
    ax = axes[1]
    se = analysis["se_matrix"]
    mask = ~np.isfinite(se)

    se_flipped = np.flipud(se)
    mask_flipped = np.flipud(mask)

    sns.heatmap(
        se_flipped,
        mask=mask_flipped,
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        xticklabels=sample_labels,
        yticklabels=oracle_labels_flipped,
        cbar_kws={"label": "Standard Error"},
        ax=ax,
    )
    ax.set_xlabel("Sample Size", fontsize=14, fontweight="bold")
    ax.set_ylabel("Oracle Coverage", fontsize=14, fontweight="bold")
    ax.set_title("B. Standard Error", fontsize=15, fontweight="bold", pad=15)

    # Panel C: MDE contours
    ax = axes[2]

    X, Y = np.meshgrid(analysis["sample_values"], analysis["oracle_values"])

    mde = analysis["mde_matrix"]
    mask = ~np.isfinite(mde)
    mde_plot = np.ma.array(mde, mask=mask)

    # Filled contours
    levels = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10, 0.20]
    cf = ax.contourf(
        X, Y, mde_plot, levels=levels, cmap="YlOrRd", extend="max"
    )
    plt.colorbar(cf, ax=ax, label="MDE (two-policy, 80% power)")

    # Key contour lines
    cs = ax.contour(
        X, Y, mde_plot,
        levels=[0.01, 0.02, 0.03, 0.05],
        colors=["black", "black", "black", "black"],
        linestyles=["--", "--", "--", "--"],
        linewidths=[2, 2, 2, 2],
    )
    ax.clabel(cs, fmt={0.01: "1%", 0.02: "2%", 0.03: "3%", 0.05: "5%"}, fontsize=10)

    # Cost contours (number of oracle labels)
    n_oracle = X * Y
    cost_lines = ax.contour(
        X, Y,
        np.ma.array(n_oracle, mask=mask),
        levels=[50, 100, 250, 500, 1000],
        colors="gray",
        linewidths=0.5,
        alpha=0.5,
    )
    ax.clabel(cost_lines, fmt=lambda v: f"{int(v)}", fontsize=8)

    ax.set_xlabel("Sample Size", fontsize=14, fontweight="bold")
    ax.set_ylabel("Oracle Coverage", fontsize=14, fontweight="bold")
    ax.set_title("C. MDE Contours", fontsize=15, fontweight="bold", pad=15)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xticks(analysis["sample_values"])
    ax.set_xticklabels([str(x) for x in analysis["sample_values"]])

    ax.set_yticks(analysis["oracle_values"])
    ax.set_yticklabels([f"{int(100*y)}%" for y in analysis["oracle_values"]])

    plt.suptitle(
        title,
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig


def main():
    """Generate MDE plot for direct+cov."""

    print("="*70)
    print("GENERATING MDE PLOT FOR DIRECT+COV")
    print("="*70)

    # Load results
    print("\nLoading results...")
    results = load_results()
    print(f"Loaded {len(results)} successful experiments")

    # Analyze direct+cov
    print("\nAnalyzing direct with covariates...")
    analysis = analyze_interaction(
        results,
        estimator="direct",
        use_covariates=True,
    )

    if not analysis["oracle_values"]:
        print("No data found for direct+cov")
        return

    # Print sweet spots
    if analysis["sweet_spots"]:
        print("\nMost cost-efficient configurations for direct+cov:")
        print("(For detecting effects with 80% power at 95% confidence)")

        for i, spot in enumerate(analysis["sweet_spots"], 1):
            print(f"\n{i}. Oracle={spot['oracle_coverage']:.1%}, n={spot['sample_size']}")
            print(f"   Oracle labels needed: {spot['n_oracle']:.0f}")
            print(f"   RMSE: {spot['rmse']:.4f}")
            print(f"   MDE: {spot['mde']:.1%}")
            print(f"   Can detect: ≥{spot['achievable_mde']:.0%} effects")
            print(f"   Cost: {spot['cost_efficiency']:.1f} labels per % MDE")

    # Create visualization
    output_path = Path("results/analysis/mde_direct_cov.png")
    create_mde_plot(
        analysis,
        "Oracle × Sample Size Analysis (DIRECT+COV)",
        output_path
    )

    print("\n" + "="*70)
    print("MDE PLOT GENERATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
