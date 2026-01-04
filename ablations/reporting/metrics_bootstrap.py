"""
Bootstrap UQ metrics for ablation analysis.

Provides metrics for evaluating uncertainty quantification methods:
- Coverage rate (primary metric)
- CI width (efficiency)
- Interval score (proper scoring rule)
- Bootstrap-specific diagnostics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Small epsilon for numerical stability
EPS = 1e-12


def coverage_rate(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Compute empirical CI coverage rate.

    Coverage = fraction of 95% CIs that contain the oracle truth.
    Target: 95% (values below indicate underconverage, above = conservative).

    Args:
        df: Tidy DataFrame with ci_covers_oracle column (bool)
        by: Grouping columns

    Returns:
        DataFrame with grouping columns + coverage_rate, coverage_se, n_experiments
    """
    def compute_coverage(group: pd.DataFrame) -> pd.Series:
        covered = group["ci_covers_oracle"].dropna()
        if len(covered) == 0:
            return pd.Series({
                "coverage_rate": np.nan,
                "coverage_se": np.nan,
                "n_experiments": 0,
            })

        n = len(covered)
        p = covered.mean()
        # SE for proportion
        se = np.sqrt(p * (1 - p) / n) if n > 1 else np.nan

        return pd.Series({
            "coverage_rate": float(p * 100),  # As percentage
            "coverage_se": float(se * 100),  # As percentage
            "n_experiments": n,
        })

    result = (
        df.groupby(by, group_keys=False)
        .apply(compute_coverage)
        .reset_index()
    )
    return result


def ci_width_stats(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Compute CI width statistics.

    Narrower CIs = more efficient (but must maintain coverage).

    Args:
        df: Tidy DataFrame with ci_width column
        by: Grouping columns

    Returns:
        DataFrame with grouping columns + ci_width_mean, ci_width_median, ci_width_std
    """
    def compute_width(group: pd.DataFrame) -> pd.Series:
        widths = group["ci_width"].dropna()
        if len(widths) == 0:
            return pd.Series({
                "ci_width_mean": np.nan,
                "ci_width_median": np.nan,
                "ci_width_std": np.nan,
            })

        return pd.Series({
            "ci_width_mean": float(widths.mean()),
            "ci_width_median": float(widths.median()),
            "ci_width_std": float(widths.std()),
        })

    result = (
        df.groupby(by, group_keys=False)
        .apply(compute_width)
        .reset_index()
    )
    return result


def interval_score(
    df: pd.DataFrame, by: List[str], alpha: float = 0.05
) -> pd.DataFrame:
    """Compute interval score (proper scoring rule).

    IS = width + (2/alpha) * (lower - oracle)_+ + (2/alpha) * (oracle - upper)_+

    Lower is better. Balances coverage (penalty) and width (sharpness).

    Args:
        df: Tidy DataFrame with ci_lo, ci_hi, oracle_truth columns (consistent with io.py)
        by: Grouping columns
        alpha: Significance level (0.05 for 95% CI)

    Returns:
        DataFrame with grouping columns + interval_score_mean
    """
    def compute_is(group: pd.DataFrame) -> pd.Series:
        scores = []
        for _, row in group.iterrows():
            # Use ci_lo/ci_hi (consistent with io.py) with fallback to ci_lower/ci_upper
            lower = row.get("ci_lo", row.get("ci_lower"))
            upper = row.get("ci_hi", row.get("ci_upper"))
            oracle = row.get("oracle_truth")

            if pd.isna(lower) or pd.isna(upper) or pd.isna(oracle):
                continue

            width = upper - lower
            penalty_low = max(0, lower - oracle) * (2 / alpha)
            penalty_high = max(0, oracle - upper) * (2 / alpha)
            score = width + penalty_low + penalty_high
            scores.append(score)

        if not scores:
            return pd.Series({
                "interval_score_mean": np.nan,
                "interval_score_median": np.nan,
            })

        return pd.Series({
            "interval_score_mean": float(np.mean(scores)),
            "interval_score_median": float(np.median(scores)),
        })

    result = (
        df.groupby(by, group_keys=False)
        .apply(compute_is)
        .reset_index()
    )
    return result


def bootstrap_diagnostics(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Compute bootstrap-specific diagnostics.

    Args:
        df: Tidy DataFrame with bootstrap metadata columns
        by: Grouping columns

    Returns:
        DataFrame with grouping columns + bootstrap metrics
    """
    def compute_diag(group: pd.DataFrame) -> pd.Series:
        # Filter to bootstrap experiments only
        boot_group = group[group["uq_method"] == "cluster_bootstrap_refit"]

        if boot_group.empty:
            return pd.Series({
                "skip_rate_mean": np.nan,
                "n_bootstrap_valid_mean": np.nan,
                "bootstrap_coupled_rate": np.nan,
            })

        # Skip rate: fraction of replicates that needed retry
        skip_rates = boot_group["bootstrap_skip_rate"].dropna()
        skip_rate_mean = skip_rates.mean() if len(skip_rates) > 0 else np.nan

        # Valid replicates
        n_valid = boot_group["n_bootstrap_valid"].dropna()
        n_valid_mean = n_valid.mean() if len(n_valid) > 0 else np.nan

        # Coupling rate
        coupled = boot_group["bootstrap_coupled"].dropna()
        coupled_rate = coupled.mean() if len(coupled) > 0 else np.nan

        return pd.Series({
            "skip_rate_mean": float(skip_rate_mean) if not pd.isna(skip_rate_mean) else np.nan,
            "n_bootstrap_valid_mean": float(n_valid_mean) if not pd.isna(n_valid_mean) else np.nan,
            "bootstrap_coupled_rate": float(coupled_rate) if not pd.isna(coupled_rate) else np.nan,
        })

    result = (
        df.groupby(by, group_keys=False)
        .apply(compute_diag)
        .reset_index()
    )
    return result


def runtime_by_method(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Compute runtime statistics by UQ method.

    Args:
        df: Tidy DataFrame with runtime_s column
        by: Grouping columns

    Returns:
        DataFrame with grouping columns + runtime_mean, runtime_median
    """
    result = (
        df.groupby(by)["runtime_s"]
        .agg(runtime_mean="mean", runtime_median="median", runtime_std="std")
        .reset_index()
    )
    return result


def compute_bootstrap_metrics(
    df: pd.DataFrame, by: List[str]
) -> pd.DataFrame:
    """Compute all bootstrap UQ metrics and merge into single DataFrame.

    Args:
        df: Tidy DataFrame with bootstrap ablation results
        by: Grouping columns (e.g., ["uq_label", "sample_size", "oracle_coverage"])

    Returns:
        DataFrame with all metrics merged on grouping columns
    """
    metrics = [
        coverage_rate(df, by),
        ci_width_stats(df, by),
        interval_score(df, by),
        runtime_by_method(df, by),
    ]

    # Add bootstrap-specific diagnostics if relevant columns exist
    if "uq_method" in df.columns:
        metrics.append(bootstrap_diagnostics(df, by))

    # Merge all metrics
    result = metrics[0]
    for metric_df in metrics[1:]:
        if not metric_df.empty:
            result = result.merge(metric_df, on=by, how="outer")

    return result


def coverage_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary table of coverage by method and regime.

    Args:
        df: Tidy DataFrame with bootstrap ablation results

    Returns:
        Pivot table with methods as columns, regimes as rows
    """
    # Compute coverage by uq_label, regime_n, regime_cov (consistent with io.py)
    by = ["uq_label", "regime_n", "regime_cov"]
    coverage = coverage_rate(df, by)

    # Create regime column
    coverage["regime"] = (
        "n=" + coverage["regime_n"].astype(str) +
        ", cov=" + (coverage["regime_cov"] * 100).astype(int).astype(str) + "%"
    )

    # Pivot to wide format
    pivot = coverage.pivot_table(
        index="regime",
        columns="uq_label",
        values="coverage_rate",
        aggfunc="first"
    ).reset_index()

    return pivot


def width_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary table of CI width by method and regime.

    Args:
        df: Tidy DataFrame with bootstrap ablation results

    Returns:
        Pivot table with methods as columns, regimes as rows
    """
    by = ["uq_label", "regime_n", "regime_cov"]
    widths = ci_width_stats(df, by)

    widths["regime"] = (
        "n=" + widths["regime_n"].astype(str) +
        ", cov=" + (widths["regime_cov"] * 100).astype(int).astype(str) + "%"
    )

    pivot = widths.pivot_table(
        index="regime",
        columns="uq_label",
        values="ci_width_mean",
        aggfunc="first"
    ).reset_index()

    return pivot
