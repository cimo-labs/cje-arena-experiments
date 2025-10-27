#!/usr/bin/env python3
"""
Interaction visualization - 2D heatmaps showing oracle coverage × sample size effects.

Adapted from legacy interaction.py to work with the new unified ablation data.
Creates heatmaps showing:
1. RMSE across oracle×sample grid
2. MDE (Minimum Detectable Effect) contours for power analysis
"""

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
    estimator: str = "stacked-dr",
    use_weight_calibration: Optional[bool] = True,
    use_iic: Optional[bool] = True,
    weight_mode: str = "hajek",
) -> Dict[str, Any]:
    """
    Analyze interaction effects between oracle coverage and sample size.

    Args:
        results: List of experiment results
        estimator: Which estimator to analyze
        use_weight_calibration: Filter to weight calibration on/off (None = both)
        use_iic: Filter to IIC on/off (None = both)
        weight_mode: Which weight mode to use

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

        # Check parameter filters
        if use_weight_calibration is not None:
            # Handle both old and new parameter names for backward compatibility
            param_value = extra.get(
                "use_weight_calibration", extra.get("use_calibration", False)
            )
            if param_value != use_weight_calibration:
                continue

        if use_iic is not None:
            if extra.get("use_iic", False) != use_iic:
                continue

        if extra.get("weight_mode", "hajek") != weight_mode:
            continue

        filtered.append(r)

    print(f"Analyzing {len(filtered)} results for {estimator}")

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

        # Get robust standard errors (includes oracle uncertainty)
        # Note: After fixing the data, robust SEs correctly equal standard SEs at 100% coverage
        if "robust_standard_errors" in r and r["robust_standard_errors"]:
            # Exclude unhelpful policy which has boundary issues
            se_dict = r["robust_standard_errors"]
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

    # Find sweet spots - configurations that achieve target MDEs efficiently
    sweet_spots = []
    target_mdes = [0.01, 0.02, 0.05]  # 1%, 2%, 5% effect sizes

    for i, oracle in enumerate(oracle_values):
        for j, n_samples in enumerate(sample_values):
            if np.isfinite(se_matrix[i, j]):
                n_oracle = oracle * n_samples
                mde = mde_two[i, j]

                # Which targets can this config achieve?
                achievable = [t for t in target_mdes if mde <= t]

                if achievable:
                    # Cost = number of oracle labels needed
                    cost_per_percent = n_oracle / (min(achievable) * 100)

                    sweet_spots.append(
                        {
                            "oracle_coverage": oracle,
                            "sample_size": n_samples,
                            "n_oracle": n_oracle,
                            "rmse": rmse_matrix[i, j],
                            "mde": mde,
                            "achievable_mde": min(achievable),
                            "cost_efficiency": cost_per_percent,
                        }
                    )

    # Sort by cost efficiency
    sweet_spots.sort(key=lambda x: x["cost_efficiency"])

    return {
        "oracle_values": oracle_values,
        "sample_values": sample_values,
        "rmse_matrix": rmse_matrix,
        "se_matrix": se_matrix,
        "mde_matrix": mde_two,
        "sweet_spots": sweet_spots[:5],  # Top 5 most efficient
    }


def create_interaction_plot(
    analysis: Dict[str, Any], estimator: str, output_path: Optional[Path] = None
) -> Optional[plt.Figure]:
    """
    Create interaction heatmaps.

    Args:
        analysis: Results from analyze_interaction
        estimator: Name of estimator for title
        output_path: Where to save the figure

    Returns:
        matplotlib Figure object or None if no data
    """

    if not analysis["oracle_values"] or not analysis["sample_values"]:
        print("No data to plot")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Common formatting
    oracle_labels = [f"{int(100*y)}%" for y in analysis["oracle_values"]]
    sample_labels = [str(x) for x in analysis["sample_values"]]

    # Panel A: RMSE heatmap
    ax = axes[0]
    rmse = analysis["rmse_matrix"]
    mask = ~np.isfinite(rmse)

    # Flip matrix vertically so oracle coverage increases upward
    rmse_flipped = np.flipud(rmse)
    mask_flipped = np.flipud(mask)
    oracle_labels_flipped = oracle_labels[::-1]

    sns.heatmap(
        rmse_flipped,
        mask=mask_flipped,
        annot=True,
        fmt=".4f",
        cmap="viridis_r",  # Light = good (low RMSE)
        vmin=0.0,
        vmax=np.nanquantile(rmse, 0.95),
        xticklabels=sample_labels,
        yticklabels=oracle_labels_flipped,
        cbar_kws={"label": "RMSE"},
        ax=ax,
    )
    ax.set_xlabel("Sample Size", fontsize=12)
    ax.set_ylabel("Oracle Coverage", fontsize=12)
    ax.set_title(f"A. RMSE ({estimator})", fontsize=13, fontweight="bold")

    # Panel B: Standard Error heatmap
    ax = axes[1]
    se = analysis["se_matrix"]
    mask = ~np.isfinite(se)

    # Flip matrix vertically so oracle coverage increases upward
    se_flipped = np.flipud(se)
    mask_flipped = np.flipud(mask)

    sns.heatmap(
        se_flipped,
        mask=mask_flipped,
        annot=True,
        fmt=".4f",
        cmap="viridis_r",  # Light = good (low SE), consistent with RMSE
        xticklabels=sample_labels,
        yticklabels=oracle_labels_flipped,  # Use the same flipped labels from above
        cbar_kws={"label": "Standard Error"},
        ax=ax,
    )
    ax.set_xlabel("Sample Size", fontsize=12)
    ax.set_ylabel("Oracle Coverage", fontsize=12)
    ax.set_title(
        "B. Standard Error (w/ Oracle Uncertainty)", fontsize=13, fontweight="bold"
    )

    # Panel C: MDE contours
    ax = axes[2]

    # Build coordinate grids
    X, Y = np.meshgrid(analysis["sample_values"], analysis["oracle_values"])

    # MDE values (masked where invalid)
    mde = analysis["mde_matrix"]
    mask = ~np.isfinite(mde)
    mde_plot = np.ma.array(mde, mask=mask)

    # Filled contours
    levels = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10, 0.20]
    cf = ax.contourf(
        X, Y, mde_plot, levels=levels, cmap="viridis_r", extend="max"
    )  # Light = good (low MDE)
    plt.colorbar(cf, ax=ax, label="MDE (two-policy, 80% power)")

    # Key contour lines at 1%, 2%, 3%, 5%
    cs = ax.contour(
        X,
        Y,
        mde_plot,
        levels=[0.01, 0.02, 0.03, 0.05],
        colors=["black", "black", "black", "black"],
        linestyles=["--", "--", "--", "--"],
        linewidths=[2, 2, 2, 2],
    )
    ax.clabel(cs, fmt={0.01: "1%", 0.02: "2%", 0.03: "3%", 0.05: "5%"}, fontsize=10)

    # Cost contours (number of oracle labels)
    n_oracle = X * Y
    cost_lines = ax.contour(
        X,
        Y,
        np.ma.array(n_oracle, mask=mask),
        levels=[50, 100, 250, 500, 1000],
        colors="gray",
        linewidths=0.5,
        alpha=0.5,
    )
    ax.clabel(cost_lines, fmt=lambda v: f"{int(v)}", fontsize=8)

    ax.set_xlabel("Sample Size", fontsize=12)
    ax.set_ylabel("Oracle Coverage", fontsize=12)
    ax.set_title("C. MDE Contours", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Set explicit x-axis ticks for sample sizes
    ax.set_xticks(analysis["sample_values"])
    ax.set_xticklabels([str(x) for x in analysis["sample_values"]])

    # Set explicit y-axis ticks and format as percentages
    ax.set_yticks(analysis["oracle_values"])
    ax.set_yticklabels([f"{int(100*y)}%" for y in analysis["oracle_values"]])

    # Don't invert - we want increasing coverage going up

    plt.suptitle(
        f"Oracle × Sample Size Interaction Analysis ({estimator.upper()})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig


def main() -> None:
    """Main analysis function."""

    print("=" * 70)
    print("INTERACTION ANALYSIS (Oracle Coverage × Sample Size)")
    print("=" * 70)

    # Load results
    print("\nLoading results...")
    results = load_results()
    print(f"Loaded {len(results)} successful experiments")

    # Analyze each estimator
    estimators = ["stacked-dr"]  # Focus on stacked-dr only

    for estimator in estimators:
        print(f"\n{'='*60}")
        print(f"Analyzing {estimator.upper()}")
        print("=" * 60)

        # Analyze with best settings for each estimator
        if estimator == "calibrated-ips":
            # Calibrated IPS is our main IPS variant
            analysis = analyze_interaction(
                results,
                estimator=estimator,
                use_weight_calibration=True,
                use_iic=False,  # IIC doesn't apply to IPS
                weight_mode="hajek",
            )
        else:
            # DR methods - don't filter by parameters
            analysis = analyze_interaction(
                results,
                estimator=estimator,
                use_weight_calibration=None,  # Don't filter
                use_iic=None,  # Don't filter
                weight_mode="hajek",
            )

        if not analysis["oracle_values"]:
            print(f"No data found for {estimator}")
            continue

        # Print sweet spots
        if analysis["sweet_spots"]:
            print(f"\nMost cost-efficient configurations for {estimator}:")
            print("(For detecting effects with 80% power at 95% confidence)")

            for i, spot in enumerate(analysis["sweet_spots"], 1):
                print(
                    f"\n{i}. Oracle={spot['oracle_coverage']:.1%}, n={spot['sample_size']}"
                )
                print(f"   Oracle labels needed: {spot['n_oracle']:.0f}")
                print(f"   RMSE: {spot['rmse']:.4f}")
                print(f"   MDE: {spot['mde']:.1%}")
                print(f"   Can detect: ≥{spot['achievable_mde']:.0%} effects")
                print(f"   Cost: {spot['cost_efficiency']:.1f} labels per % MDE")

        # Create visualization
        output_path = Path(f"results/analysis/interaction_{estimator}.png")
        create_interaction_plot(analysis, estimator, output_path)

    # Also create comparison plot for DR methods with/without enhancements
    print("\n" + "=" * 60)
    print("DR ENHANCEMENT COMPARISON (Standard Errors)")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    configs = [
        ("No Enhancements", False, False),
        ("Calibration Only", True, False),
        ("IIC Only", False, True),
        ("Full Enhancement", True, True),
    ]

    for idx, (title, use_cal, use_iic) in enumerate(configs):
        ax = axes[idx // 2, idx % 2]

        analysis = analyze_interaction(
            results,
            estimator="stacked-dr",
            use_weight_calibration=use_cal,
            use_iic=use_iic,
            weight_mode="hajek",
        )

        if analysis["oracle_values"]:
            se = analysis["se_matrix"]  # Use standard errors instead of RMSE
            mask = ~np.isfinite(se)

            # Flip matrix vertically so oracle coverage increases upward
            se_flipped = np.flipud(se)
            mask_flipped = np.flipud(mask)
            oracle_labels = [f"{int(100*y)}%" for y in analysis["oracle_values"]]
            oracle_labels_flipped = oracle_labels[::-1]

            sns.heatmap(
                se_flipped,
                mask=mask_flipped,
                annot=True,
                fmt=".4f",
                cmap="viridis_r",  # Consistent colormap: light = good (low SE)
                vmin=0.0,
                vmax=0.02,  # Fixed scale for SE comparison
                xticklabels=[str(x) for x in analysis["sample_values"]],
                yticklabels=oracle_labels_flipped,
                cbar_kws={"label": "Standard Error"},
                ax=ax,
            )
            ax.set_xlabel("Sample Size")
            ax.set_ylabel("Oracle Coverage")
            ax.set_title(f"{title}\n(Cal={use_cal}, IIC={use_iic})")

    plt.suptitle(
        "DR Enhancement Impact on Standard Errors (Oracle×Sample Interaction)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        "results/analysis/interaction_dr_comparison.png", dpi=150, bbox_inches="tight"
    )
    print("Saved DR comparison to results/analysis/interaction_dr_comparison.png")

    print("\n" + "=" * 70)
    print("INTERACTION ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
