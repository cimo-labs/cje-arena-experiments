#!/usr/bin/env python
"""
Unified CLI for generating all paper tables and figures.

Usage:
    python -m reporting.cli_generate \
        --results results/all_experiments.jsonl \
        --output tables/ \
        --format latex
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
import numpy as np

# Suppress harmless numpy warnings about empty slices
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*Mean of empty slice.*"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*Degrees of freedom <= 0.*"
)

# Suppress pandas FutureWarnings about groupby behavior (will be addressed when upgrading pandas)
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*DataFrameGroupBy.apply.*"
)

# Suppress regex pattern warnings that don't affect functionality
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*pattern is interpreted as a regular expression.*",
)

# Import all modules
from . import io
from . import tables_main
from . import format_latex


def parse_regimes(regime_str: str) -> List[Tuple[int, float]]:
    """Parse regime specification string.

    Args:
        regime_str: String like "250,500;0.05,0.10|1000,2500;0.25,0.50"

    Returns:
        List of (sample_size, coverage) tuples
    """
    regimes: List[Tuple[int, float]] = []

    if not regime_str:
        return regimes

    # Split by | to get regime groups
    groups = regime_str.split("|")

    for group in groups:
        # Split by ; to separate sample sizes and coverages
        parts = group.split(";")
        if len(parts) != 2:
            warnings.warn(f"Invalid regime group: {group}")
            continue

        # Parse sample sizes
        sample_sizes = [int(n.strip()) for n in parts[0].split(",")]

        # Parse coverages
        coverages = [float(c.strip()) for c in parts[1].split(",")]

        # Create all combinations
        for n in sample_sizes:
            for cov in coverages:
                regimes.append((n, cov))

    return regimes


def main() -> int:
    """Main entry point for table generation."""
    parser = argparse.ArgumentParser(
        description="Generate paper tables from experiment results"
    )

    parser.add_argument(
        "--results", type=Path, required=True, help="Path to JSONL results file"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tables/"),
        help="Output directory for tables",
    )

    parser.add_argument(
        "--format",
        choices=["latex", "markdown", "both"],
        default="latex",
        help="Output format",
    )

    parser.add_argument(
        "--regimes",
        type=str,
        default="",
        help="Regime specification (e.g., '250,500;0.05,0.10|1000,2500;0.25,0.50')",
    )

    # Changed to store_false with new name so default behavior includes unhelpful
    parser.add_argument(
        "--exclude-unhelpful",
        action="store_true",
        help="Exclude unhelpful policy from ranking metrics (not recommended)",
    )

    parser.add_argument(
        "--tables",
        type=str,
        default="all",
        help="Comma-separated list of tables to generate (m1,m2,m3,boundary or 'all')",
    )

    parser.add_argument(
        "--show-regimes",
        action="store_true",
        help="Show per-regime breakdowns in Table M1 (quadrants)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Validate inputs
    if not args.results.exists():
        print(f"Error: Results file not found: {args.results}")
        return 1

    # Create output directories
    output_main = args.output / "main"
    output_appendix = args.output / "appendix"
    output_figures = args.output / "figures"
    output_quadrant = args.output / "quadrant"

    output_main.mkdir(parents=True, exist_ok=True)
    output_appendix.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)
    output_quadrant.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.verbose:
        print(f"Loading results from {args.results}...")

    # Default behavior: include unhelpful (for correct ranking metrics)
    # Only exclude if explicitly requested
    df = io.load_results_jsonl(
        args.results, include_unhelpful=not args.exclude_unhelpful
    )

    if df.empty:
        print("Warning: No valid results found in input file")
        return 1

    # Deduplicate
    df = io.deduplicate_runs(df)

    if args.verbose:
        print(f"Loaded {len(df)} records")
        print(f"Estimators: {df['estimator'].unique()}")
        print(f"Regimes: {df[['regime_n', 'regime_cov']].drop_duplicates()}")

    # Parse regimes if specified
    regimes = parse_regimes(args.regimes) if args.regimes else None

    # Filter to specified regimes
    if regimes:
        if args.verbose:
            print(f"Filtering to regimes: {regimes}")
        df = io.filter_by_regime(
            df, sample_sizes=[r[0] for r in regimes], coverages=[r[1] for r in regimes]
        )

    # Determine which tables to generate
    if args.tables == "all":
        tables_to_generate = ["m1", "m2", "m3", "boundary", "quadrants"]
    else:
        tables_to_generate = [t.strip().lower() for t in args.tables.split(",")]

    # Generate tables
    generated_files = []

    # Table M1: Accuracy by Regime
    if "m1" in tables_to_generate:
        if args.verbose:
            print("Generating Table M1: Accuracy by Regime...")

        table_m1 = tables_main.build_table_m1_accuracy_by_regime(
            df, regimes=regimes, include_overall=True, show_regimes=args.show_regimes
        )

        if args.format in ["latex", "both"]:
            latex_m1 = format_latex.format_table_m1(table_m1)
            output_file = output_main / "table_m1_accuracy.tex"
            output_file.write_text(latex_m1)
            generated_files.append(str(output_file))

        if args.format in ["markdown", "both"]:
            md_m1 = table_m1.to_markdown(index=False, floatfmt=".4f")
            # Ensure proper line breaks in markdown
            md_m1 = md_m1.replace("\n", "\n")  # Ensure Unix line endings
            output_file = output_main / "table_m1_accuracy.md"
            output_file.write_text(md_m1 + "\n")  # Add final newline
            generated_files.append(str(output_file))

    # Table M2: Design Deltas
    if "m2" in tables_to_generate:
        if args.verbose:
            print("Generating Table M2: ESS Comparison (Weight Calibration Effect)...")

        # Generate ESS comparison table
        table_m2_ess = tables_main.build_table_ess_comparison(df)

        # Only generate output if we have data
        if not table_m2_ess.empty:
            if args.format in ["latex", "both"]:
                latex_m2 = format_latex.format_table_ess_comparison(table_m2_ess)
                output_file = output_main / "table_m2_ess_comparison.tex"
                output_file.write_text(latex_m2)
                generated_files.append(str(output_file))

            if args.format in ["markdown", "both"]:
                md_m2 = "# Table M2: ESS Comparison (Weight Calibration Effect)\n\n"
                md_m2 += "Shows the dramatic improvement in Effective Sample Size from SIMCal weight calibration.\n\n"
                md_m2 += table_m2_ess.to_markdown(index=False)
                output_file = output_main / "table_m2_ess_comparison.md"
                output_file.write_text(md_m2)
                generated_files.append(str(output_file))
        elif args.verbose:
            print("  Skipping Table M2 (no IPS data available yet)")

    # Table M3: Gates
    if "m3" in tables_to_generate:
        if args.verbose:
            print("Generating Table M3: Gates & Diagnostics...")

        table_m3 = tables_main.build_table_m3_gates(
            df, by_regime=False  # Set to True for regime-specific breakdown
        )

        if args.format in ["latex", "both"]:
            latex_m3 = format_latex.format_table_m3_gates(table_m3)
            output_file = output_main / "table_m3_gates.tex"
            output_file.write_text(latex_m3)
            generated_files.append(str(output_file))

        if args.format in ["markdown", "both"]:
            md_m3 = table_m3.to_markdown(index=False, floatfmt=".2f")
            output_file = output_main / "table_m3_gates.md"
            output_file.write_text(md_m3)
            generated_files.append(str(output_file))

    # Boundary Refuse Rates Table
    if "boundary" in tables_to_generate or "refuse" in tables_to_generate:
        if args.verbose:
            print("Generating Boundary Refuse Rates Table...")

        table_refuse = tables_main.build_table_refuse_rates(df)

        if args.format in ["latex", "both"]:
            latex_refuse = format_latex.format_table_refuse_rates(table_refuse)
            output_file = output_main / "table_refuse_rates.tex"
            output_file.write_text(latex_refuse)
            generated_files.append(str(output_file))

        if args.format in ["markdown", "both"]:
            md_refuse = table_refuse.to_markdown(index=False)
            output_file = output_main / "table_refuse_rates.md"
            output_file.write_text(md_refuse)
            generated_files.append(str(output_file))

    # Generate quadrant leaderboards
    if "quadrants" in tables_to_generate:
        if args.verbose:
            print("Generating Quadrant Leaderboards...")

        quadrant_tables = tables_main.build_quadrant_leaderboards(df)

        for quad_name, quad_df in quadrant_tables.items():
            safe_name = quad_name.lower().replace(" ", "_").replace("-", "_")

            if args.format in ["latex", "both"]:
                latex_quad = format_latex.format_table_m1(
                    quad_df,
                    caption=f"{quad_name} Leaderboard",
                    label=f"tab:leaderboard-{safe_name}",
                )
                output_file = output_quadrant / f"leaderboard_{safe_name}.tex"
                output_file.write_text(latex_quad)
                generated_files.append(str(output_file))

            if args.format in ["markdown", "both"]:
                md_quad = f"## {quad_name} Leaderboard\n\n"
                md_quad += quad_df.to_markdown(index=False, floatfmt=".4f")
                output_file = output_quadrant / f"leaderboard_{safe_name}.md"
                output_file.write_text(md_quad)
                generated_files.append(str(output_file))

    # Generate figure data
    if args.verbose:
        print("Generating figure data...")

    fig_data = tables_main.build_figure_m1_coverage_vs_width_data(df)
    fig_data_file = output_figures / "coverage_vs_width_data.csv"
    fig_data.to_csv(fig_data_file, index=False)
    generated_files.append(str(fig_data_file))

    # Generate summary statistics
    if args.verbose:
        print("Generating summary statistics...")

    summary = tables_main.build_summary_statistics(df, output_format="text")
    summary_file = args.output / "summary_statistics.txt"
    summary_file.write_text(summary)
    generated_files.append(str(summary_file))

    # Print results
    print(f"\nGenerated {len(generated_files)} files:")
    for f in generated_files:
        print(f"  - {f}")

    return 0


if __name__ == "__main__":
    exit(main())
