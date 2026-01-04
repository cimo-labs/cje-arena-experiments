#!/usr/bin/env python3
"""
Table generators for Bootstrap UQ ablation study.

Generates comparison tables for the three UQ methods:
- Table B1: Coverage Comparison by Method
- Table B2: CI Width Comparison
- Table B3: Bootstrap Parameter Sensitivity

Uses the standard io.py loader pattern with bootstrap-specific extensions.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse

from .metrics_bootstrap import (
    coverage_rate,
    ci_width_stats,
    interval_score,
    runtime_by_method,
    compute_bootstrap_metrics,
)


# Bootstrap UQ policies (same as default, exclude unhelpful)
BOOTSTRAP_POLICIES = ("clone", "parallel_universe_prompt", "premium")


def load_bootstrap_results(
    path: Path,
    policies: Tuple[str, ...] = BOOTSTRAP_POLICIES,
    include_unhelpful: bool = False,
) -> pd.DataFrame:
    """Load bootstrap UQ ablation results into tidy DataFrame.

    Follows the io.load_results_jsonl pattern but with bootstrap-specific fields.

    Args:
        path: Path to JSONL results file
        policies: Policies to include (default: well-behaved only)
        include_unhelpful: Whether to include the unhelpful policy

    Returns:
        Tidy DataFrame with columns matching existing infrastructure plus bootstrap fields
    """
    if include_unhelpful:
        policies = policies + ("unhelpful",)

    rows = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {line_num}: {e}")
                continue

            if not r.get("success"):
                continue

            spec = r.get("spec", {})
            extra = spec.get("extra", {})

            # Base fields (consistent with io.py naming)
            base_fields = {
                "run_id": f"bootstrap_uq_{line_num}",
                "seed": spec.get("seed_base", r.get("seed")),
                "regime_n": spec.get("sample_size"),  # Use regime_n for consistency
                "regime_cov": spec.get("oracle_coverage"),  # Use regime_cov for consistency
                "estimator": spec.get("estimator"),
                "runtime_s": r.get("runtime_s"),
                # Bootstrap-specific fields
                "uq_method_requested": extra.get("inference_method"),
                "oua_jackknife": extra.get("oua_jackknife"),
                "n_bootstrap_requested": extra.get("n_bootstrap"),
                "uq_label": extra.get("uq_label"),
                "uq_method": r.get("uq_method"),
                "n_bootstrap_valid": r.get("n_bootstrap_valid"),
                "bootstrap_skip_rate": r.get("bootstrap_skip_rate"),
                "bootstrap_coupled": r.get("bootstrap_coupled"),
            }

            # Get metric dictionaries
            estimates = r.get("estimates", {})
            oracle_truths = r.get("oracle_truths", {})
            cis = r.get("confidence_intervals", {})
            ci_covers = r.get("ci_covers_oracle", {})

            for pol in policies:
                if pol not in estimates:
                    continue

                # Extract CIs (consistent with io.py)
                ci = cis.get(pol, [None, None])
                if ci and ci[0] is not None and ci[1] is not None:
                    ci = sorted(ci)
                else:
                    ci = [None, None]

                row = {
                    **base_fields,
                    "policy": pol,
                    "est": estimates.get(pol),
                    "oracle_truth": oracle_truths.get(pol),
                    "ci_lo": ci[0],
                    "ci_hi": ci[1],
                    # Precomputed coverage (also compute from CIs for consistency)
                    "ci_covers_oracle": ci_covers.get(pol),
                }
                rows.append(row)

    df = pd.DataFrame(rows)

    # Add derived columns (consistent with io.py)
    if not df.empty:
        # CI width
        df["ci_width"] = df["ci_hi"] - df["ci_lo"]

        # Coverage indicator (compute from CIs as backup)
        df["covered"] = (
            (df["ci_lo"].fillna(float('inf')) <= df["oracle_truth"]) &
            (df["oracle_truth"] <= df["ci_hi"].fillna(float('-inf')))
        )

        # Use precomputed ci_covers_oracle if available, else use computed
        df["ci_covers_oracle"] = df["ci_covers_oracle"].fillna(df["covered"])

        # Error metrics
        df["error"] = df["est"] - df["oracle_truth"]
        df["abs_error"] = df["error"].abs()

    return df


def generate_coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate Table B1: Coverage Comparison by Method.

    Rows: Sample size x Oracle coverage combinations
    Columns: UQ methods (cluster_robust, oua_jackknife, bootstrap variants)

    Args:
        df: Tidy DataFrame from load_bootstrap_results

    Returns:
        Pivot table with coverage rates
    """
    # Group by method and regime (use regime_n, regime_cov for consistency)
    by = ["uq_label", "regime_n", "regime_cov"]
    coverage = coverage_rate(df, by)

    # Create regime label
    coverage["regime"] = (
        "n=" + coverage["regime_n"].astype(str) +
        ", cov=" + (coverage["regime_cov"] * 100).astype(int).astype(str) + "%"
    )

    # Pivot to wide format
    pivot = coverage.pivot_table(
        index=["regime_n", "regime_cov", "regime"],
        columns="uq_label",
        values="coverage_rate",
        aggfunc="first"
    ).reset_index()

    # Sort by regime_n and regime_cov
    pivot = pivot.sort_values(["regime_n", "regime_cov"])

    # Reorder columns
    method_cols = [c for c in pivot.columns if c not in ["regime_n", "regime_cov", "regime"]]
    desired_order = ["cluster_robust", "oua_jackknife"]
    desired_order += [c for c in sorted(method_cols) if c.startswith("bootstrap")]
    method_cols = [c for c in desired_order if c in method_cols]

    return pivot[["regime"] + method_cols]


def generate_width_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate Table B2: CI Width Comparison.

    Args:
        df: Tidy DataFrame from load_bootstrap_results

    Returns:
        Pivot table with CI widths
    """
    by = ["uq_label", "regime_n", "regime_cov"]
    widths = ci_width_stats(df, by)

    widths["regime"] = (
        "n=" + widths["regime_n"].astype(str) +
        ", cov=" + (widths["regime_cov"] * 100).astype(int).astype(str) + "%"
    )

    pivot = widths.pivot_table(
        index=["regime_n", "regime_cov", "regime"],
        columns="uq_label",
        values="ci_width_mean",
        aggfunc="first"
    ).reset_index()

    pivot = pivot.sort_values(["regime_n", "regime_cov"])

    method_cols = [c for c in pivot.columns if c not in ["regime_n", "regime_cov", "regime"]]
    desired_order = ["cluster_robust", "oua_jackknife"]
    desired_order += [c for c in sorted(method_cols) if c.startswith("bootstrap")]
    method_cols = [c for c in desired_order if c in method_cols]

    return pivot[["regime"] + method_cols]


def generate_bootstrap_sensitivity_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate Table B3: Bootstrap Parameter Sensitivity.

    Shows coverage, width, and runtime by n_bootstrap value.

    Args:
        df: Tidy DataFrame from load_results

    Returns:
        Table with bootstrap parameter comparison
    """
    # Filter to bootstrap experiments only
    boot_df = df[df["uq_label"].str.startswith("bootstrap", na=False)]

    if boot_df.empty:
        return pd.DataFrame()

    # Group by n_bootstrap
    by = ["n_bootstrap_requested"]
    coverage = coverage_rate(boot_df, by)
    widths = ci_width_stats(boot_df, by)
    runtime = runtime_by_method(boot_df, by)

    # Merge metrics
    result = coverage.merge(widths, on=by, how="outer")
    result = result.merge(runtime, on=by, how="outer")

    # Rename columns for clarity
    result = result.rename(columns={
        "n_bootstrap_requested": "n_bootstrap",
        "coverage_rate": "Coverage (%)",
        "ci_width_mean": "CI Width",
        "runtime_mean": "Runtime (s)",
    })

    # Select and order columns
    cols = ["n_bootstrap", "Coverage (%)", "CI Width", "Runtime (s)"]
    result = result[[c for c in cols if c in result.columns]]

    return result.sort_values("n_bootstrap")


def generate_interval_score_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate table with interval scores by method and regime.

    Interval score is a proper scoring rule that balances coverage and width.

    Args:
        df: Tidy DataFrame from load_bootstrap_results

    Returns:
        Pivot table with interval scores
    """
    by = ["uq_label", "regime_n", "regime_cov"]
    scores = interval_score(df, by)

    scores["regime"] = (
        "n=" + scores["regime_n"].astype(str) +
        ", cov=" + (scores["regime_cov"] * 100).astype(int).astype(str) + "%"
    )

    pivot = scores.pivot_table(
        index=["regime_n", "regime_cov", "regime"],
        columns="uq_label",
        values="interval_score_mean",
        aggfunc="first"
    ).reset_index()

    pivot = pivot.sort_values(["regime_n", "regime_cov"])

    method_cols = [c for c in pivot.columns if c not in ["regime_n", "regime_cov", "regime"]]
    desired_order = ["cluster_robust", "oua_jackknife"]
    desired_order += [c for c in sorted(method_cols) if c.startswith("bootstrap")]
    method_cols = [c for c in desired_order if c in method_cols]

    return pivot[["regime"] + method_cols]


def format_latex_coverage(table: pd.DataFrame) -> str:
    """Format coverage table as LaTeX.

    Highlights values outside 93-97% range.

    Args:
        table: Coverage table from generate_coverage_table

    Returns:
        LaTeX string
    """
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Coverage Comparison by UQ Method (target: 95\%)}")
    lines.append(r"\label{tab:bootstrap_coverage}")

    # Header
    cols = [c for c in table.columns if c != "regime"]
    header = " & ".join(["Regime"] + cols) + r" \\"
    lines.append(r"\begin{tabular}{l" + "r" * len(cols) + "}")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")

    # Rows
    for _, row in table.iterrows():
        cells = [row["regime"]]
        for col in cols:
            val = row.get(col)
            if pd.isna(val):
                cells.append("--")
            else:
                # Highlight under/over coverage
                if val < 93:
                    cells.append(r"\textcolor{red}{" + f"{val:.1f}" + "}")
                elif val > 97:
                    cells.append(r"\textcolor{blue}{" + f"{val:.1f}" + "}")
                else:
                    cells.append(f"{val:.1f}")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def format_latex_width(table: pd.DataFrame) -> str:
    """Format CI width table as LaTeX.

    Args:
        table: Width table from generate_width_table

    Returns:
        LaTeX string
    """
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{CI Width Comparison by UQ Method}")
    lines.append(r"\label{tab:bootstrap_width}")

    cols = [c for c in table.columns if c != "regime"]
    header = " & ".join(["Regime"] + cols) + r" \\"
    lines.append(r"\begin{tabular}{l" + "r" * len(cols) + "}")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")

    for _, row in table.iterrows():
        cells = [row["regime"]]
        for col in cols:
            val = row.get(col)
            if pd.isna(val):
                cells.append("--")
            else:
                cells.append(f"{val:.4f}")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    """Generate all bootstrap UQ tables from results file."""
    parser = argparse.ArgumentParser(
        description="Generate Bootstrap UQ comparison tables"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="results/bootstrap_uq.jsonl",
        help="Path to results JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tables/bootstrap_uq/",
        help="Output directory for tables",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "latex", "both"],
        default="both",
        help="Output format",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_path}...")
    df = load_bootstrap_results(results_path)
    print(f"Loaded {len(df)} records")

    # Generate tables
    print("Generating coverage table (B1)...")
    coverage_table = generate_coverage_table(df)

    print("Generating width table (B2)...")
    width_table = generate_width_table(df)

    print("Generating bootstrap sensitivity table (B3)...")
    sensitivity_table = generate_bootstrap_sensitivity_table(df)

    print("Generating interval score table...")
    is_table = generate_interval_score_table(df)

    # Save tables
    if args.format in ["csv", "both"]:
        coverage_table.to_csv(output_dir / "B1_coverage.csv", index=False)
        width_table.to_csv(output_dir / "B2_width.csv", index=False)
        sensitivity_table.to_csv(output_dir / "B3_bootstrap_sensitivity.csv", index=False)
        is_table.to_csv(output_dir / "interval_score.csv", index=False)
        print(f"CSV tables saved to {output_dir}")

    if args.format in ["latex", "both"]:
        with open(output_dir / "B1_coverage.tex", "w") as f:
            f.write(format_latex_coverage(coverage_table))
        with open(output_dir / "B2_width.tex", "w") as f:
            f.write(format_latex_width(width_table))
        print(f"LaTeX tables saved to {output_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if not coverage_table.empty:
        print("\nCoverage by Method (aggregated):")
        method_cols = [c for c in coverage_table.columns if c != "regime"]
        for col in method_cols:
            vals = coverage_table[col].dropna()
            if len(vals) > 0:
                print(f"  {col}: {vals.mean():.1f}% (range: {vals.min():.1f}-{vals.max():.1f}%)")

    if not sensitivity_table.empty:
        print("\nBootstrap Parameter Sensitivity:")
        print(sensitivity_table.to_string(index=False))

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
