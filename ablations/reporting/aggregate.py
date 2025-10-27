"""
Aggregation utilities for experiment results.

Includes groupby wrappers, paired delta computation, and bootstrap utilities.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Callable
from scipy import stats
import warnings

from . import metrics


def by_regime(df: pd.DataFrame, include_mc: bool = True) -> pd.DataFrame:
    """Aggregate metrics by (estimator, regime_n, regime_cov).

    Args:
        df: Tidy DataFrame from io.load_results_jsonl
        include_mc: Whether to include MC variance diagnostics

    Returns:
        DataFrame with metrics aggregated by regime
    """
    by_cols = ["estimator", "regime_n", "regime_cov"]
    result = metrics.compute_all_metrics(df, by_cols, include_debiased=True)

    # Add display names if available
    if "estimator_display" in df.columns:
        display_map = df.groupby("estimator")["estimator_display"].first().to_dict()
        result["estimator_display"] = result["estimator"].map(display_map).fillna(result["estimator"])

    return result


def by_estimator(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by estimator only (averaging across regimes).

    Args:
        df: Tidy DataFrame from io.load_results_jsonl

    Returns:
        DataFrame with metrics aggregated by estimator
    """
    by_cols = ["estimator"]
    result = metrics.compute_all_metrics(df, by_cols, include_debiased=True)

    # Add display names if available
    if "estimator_display" in df.columns:
        display_map = df.groupby("estimator")["estimator_display"].first().to_dict()
        result["estimator_display"] = result["estimator"].map(display_map).fillna(result["estimator"])

    return result


def by_estimator_and_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by (estimator, seed) for paired comparisons.

    Args:
        df: Tidy DataFrame from io.load_results_jsonl

    Returns:
        DataFrame with metrics aggregated by estimator and seed
    """
    by_cols = ["estimator", "seed"]
    return metrics.compute_all_metrics(df, by_cols, include_debiased=True)


def paired_delta(
    df: pd.DataFrame,
    toggle: str = "use_calib",
    match_on: Tuple[str, ...] = ("estimator", "regime_n", "regime_cov", "seed"),
    metrics_to_compare: Optional[List[str]] = None,
    bootstrap_n: int = 1000,
    aggregate_across_regimes: bool = True,
) -> pd.DataFrame:
    """Compute paired differences for a design choice toggle.

    Args:
        df: Tidy DataFrame
        toggle: Column name to toggle (e.g., 'use_calib', 'outer_cv')
        match_on: Columns that must match for pairing
        metrics_to_compare: List of metric columns to compute deltas for
        bootstrap_n: Number of bootstrap samples for CIs
        aggregate_across_regimes: If True, aggregate deltas across regimes

    Returns:
        DataFrame with delta statistics and significance tests
    """
    if metrics_to_compare is None:
        metrics_to_compare = [
            "rmse_d",
            "interval_score_oa",
            "calib_score",
            "se_geomean",
            "kendall_tau",
            "coverage_robust",
        ]

    # First compute metrics at appropriate level
    by_cols = list(match_on) + [toggle]
    df_metrics = metrics.compute_all_metrics(df, by_cols)

    # Separate on/off groups
    df_on = df_metrics[df_metrics[toggle] == True].copy()
    df_off = df_metrics[df_metrics[toggle] == False].copy()

    # Merge to find pairs
    merge_cols = [c for c in match_on if c in df_metrics.columns]
    df_paired = df_on.merge(
        df_off, on=merge_cols, suffixes=("_on", "_off"), how="inner"
    )

    if df_paired.empty:
        warnings.warn(f"No paired observations found for toggle '{toggle}'")
        return pd.DataFrame()

    # Compute deltas for each metric
    delta_results = []

    # Determine grouping columns based on aggregate_across_regimes
    if aggregate_across_regimes:
        # Only group by estimator, collecting all deltas across regimes
        group_cols = ["estimator"] if "estimator" in merge_cols else []
    else:
        # Group by estimator and regime
        group_cols = [
            c for c in ["estimator", "regime_n", "regime_cov"] if c in merge_cols
        ]

    if not group_cols:
        group_cols = ["estimator"]  # Fallback

    for group_vals, group_df in df_paired.groupby(group_cols):
        # Handle groupby results - always returns tuple when list is provided
        if len(group_cols) == 1:
            # Single column groupby with list still returns tuple
            row = {
                group_cols[0]: (
                    group_vals[0] if isinstance(group_vals, tuple) else group_vals
                )
            }
        else:
            row = dict(zip(group_cols, group_vals))
        row["n_pairs"] = len(group_df)

        for metric in metrics_to_compare:
            col_on = f"{metric}_on"
            col_off = f"{metric}_off"

            if col_on not in group_df.columns or col_off not in group_df.columns:
                continue

            # Get valid pairs
            valid_mask = group_df[col_on].notna() & group_df[col_off].notna()
            if not valid_mask.any():
                continue

            vals_on = group_df.loc[valid_mask, col_on].values
            vals_off = group_df.loc[valid_mask, col_off].values
            deltas = vals_on - vals_off

            if len(deltas) == 0:
                continue

            # Compute statistics
            delta_mean = np.mean(deltas)
            delta_se = (
                np.std(deltas) / np.sqrt(len(deltas)) if len(deltas) > 1 else np.nan
            )

            # Wilcoxon signed-rank test
            if len(deltas) >= 6 and len(set(deltas)) > 1:
                try:
                    _, p_value = stats.wilcoxon(deltas, alternative="two-sided")
                except:
                    p_value = np.nan
            else:
                p_value = np.nan

            # Bootstrap CI with reproducible seed
            if len(deltas) >= 20 and bootstrap_n > 0:
                ci_low, ci_high = bootstrap_ci(deltas, n_samples=bootstrap_n, seed=42)
            else:
                ci_low, ci_high = np.nan, np.nan

            # Add to results
            row[f"Δ{metric}"] = delta_mean
            row[f"Δ{metric}_se"] = delta_se
            row[f"Δ{metric}_p"] = p_value
            row[f"Δ{metric}_ci_low"] = ci_low
            row[f"Δ{metric}_ci_high"] = ci_high

            # Percent change for SE (mean of per-pair percentage changes)
            if metric == "se_geomean":
                denom = np.where(vals_off == 0, np.nan, vals_off)
                pct = (vals_on - vals_off) / denom * 100.0
                row[f"Δ{metric}_pct"] = float(np.nanmean(pct))

        delta_results.append(row)

    return pd.DataFrame(delta_results)


def bootstrap_ci(
    data: np.ndarray,
    n_samples: int = 1000,
    confidence: float = 0.95,
    statistic: Callable = np.mean,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval.

    Args:
        data: 1D array of values
        n_samples: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        statistic: Function to compute statistic (default: mean)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (lower, upper) confidence bounds
    """
    if len(data) == 0:
        return np.nan, np.nan

    # Generate bootstrap samples with optional seed
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    bootstrap_stats = []
    n = len(data)

    for _ in range(n_samples):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))

    # Compute percentiles
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return float(lower), float(upper)


def summarize_across_seeds(
    df_metrics: pd.DataFrame,
    group_by: List[str],
    metric_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Summarize metrics across seeds with mean and SE.

    Args:
        df_metrics: DataFrame with metrics (e.g., from by_regime)
        group_by: Columns to group by (e.g., ["estimator", "regime_n", "regime_cov"])
        metric_cols: Columns to summarize (None = auto-detect)

    Returns:
        DataFrame with mean and SE for each metric
    """
    if metric_cols is None:
        # Auto-detect numeric columns that aren't grouping columns
        metric_cols = [
            c
            for c in df_metrics.columns
            if c not in group_by and pd.api.types.is_numeric_dtype(df_metrics[c])
        ]

    results = []
    for group_vals, group_df in df_metrics.groupby(group_by):
        row = (
            dict(zip(group_by, group_vals))
            if len(group_by) > 1
            else {group_by[0]: group_vals}
        )

        for metric in metric_cols:
            if metric not in group_df.columns:
                continue

            values = group_df[metric].dropna()
            if len(values) == 0:
                continue

            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_se"] = (
                values.std() / np.sqrt(len(values)) if len(values) > 1 else np.nan
            )
            row[f"{metric}_n"] = len(values)

        results.append(row)

    return pd.DataFrame(results)


def compute_variance_cap_sensitivity(
    df: pd.DataFrame,
    rho_values: List[float] = [1.0, 2.0],
    metrics_to_check: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute sensitivity to variance cap parameter using paired matching.

    Args:
        df: Tidy DataFrame with rho column
        rho_values: Values of rho to compare
        metrics_to_check: Metrics to check for sensitivity

    Returns:
        DataFrame with max absolute changes across rho values
    """
    if metrics_to_check is None:
        metrics_to_check = ["rmse_d", "se_geomean", "coverage_robust"]

    # Check if rho column exists
    if "rho" not in df.columns:
        warnings.warn("No 'rho' column in data - skipping variance cap sensitivity")
        return pd.DataFrame()

    # Filter to relevant rho values
    df_rho = df[df["rho"].isin(rho_values)]

    if df_rho.empty:
        warnings.warn(f"No data found for rho values {rho_values}")
        return pd.DataFrame()

    # Compute metrics for each rho
    by_cols = ["estimator", "regime_n", "regime_cov", "seed", "rho"]
    df_metrics = metrics.compute_all_metrics(df_rho, by_cols)

    rows = []
    for est in df_metrics["estimator"].dropna().unique():
        row = {"estimator": est}  # Use lowercase for consistency
        df_e = df_metrics[df_metrics["estimator"] == est].copy()

        for m in metrics_to_check:
            if m not in df_e.columns:
                continue

            # Pivot to wide format for paired comparison
            keys = ["regime_n", "regime_cov", "seed"]
            try:
                wide = df_e.pivot_table(
                    index=keys, columns="rho", values=m, aggfunc="first"
                )
            except:
                continue

            # Ensure both rho columns present
            if not set(rho_values).issubset(set(wide.columns)):
                continue

            # Compute paired differences
            diffs = wide[rho_values[1]] - wide[rho_values[0]]
            max_abs = float(np.nanmax(np.abs(diffs.values))) if diffs.size else np.nan

            # Format result
            if not np.isnan(max_abs):
                if max_abs < 1e-4:
                    row[f"Max |Δ{m}|"] = "< 1e-4"
                else:
                    row[f"Max |Δ{m}|"] = f"{max_abs:.4f}"
            else:
                row[f"Max |Δ{m}|"] = None

        rows.append(row)

    out = pd.DataFrame(rows)
    # Keep only populated columns
    if not out.empty:
        out = out.loc[:, ~out.isna().all()]
    return out


def compute_outlier_robust_bounds(
    df_metrics: pd.DataFrame,
    metric_cols: Optional[List[str]] = None,
    method: str = "iqr",
    iqr_multiplier: float = 1.5,
) -> Dict[str, Tuple[float, float]]:
    """Compute robust normalization bounds for metrics.

    Args:
        df_metrics: DataFrame with metric columns
        metric_cols: Columns to compute bounds for
        method: 'iqr' for IQR-based bounds, 'percentile' for percentile-based
        iqr_multiplier: Multiplier for IQR fence

    Returns:
        Dict mapping metric names to (min, max) bounds
    """
    if metric_cols is None:
        # Auto-detect numeric columns
        metric_cols = [
            c
            for c in df_metrics.columns
            if pd.api.types.is_numeric_dtype(df_metrics[c])
            and not c.startswith("n_")
            and not c.endswith("_n")
        ]

    bounds = {}

    for col in metric_cols:
        if col not in df_metrics.columns:
            continue

        values = df_metrics[col].dropna().values
        if len(values) == 0:
            bounds[col] = (0.0, 1.0)
            continue

        if method == "iqr" and len(values) >= 4:
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower_fence = q1 - iqr_multiplier * iqr
            upper_fence = q3 + iqr_multiplier * iqr
            capped_values = np.clip(values, lower_fence, upper_fence)
            bounds[col] = (float(capped_values.min()), float(capped_values.max()))

        elif method == "percentile":
            bounds[col] = (
                float(np.percentile(values, 5)),
                float(np.percentile(values, 95)),
            )
        else:
            bounds[col] = (float(values.min()), float(values.max()))

    # Add fixed bounds for known metrics
    if "kendall_tau" in metric_cols:
        bounds["kendall_tau"] = (-1.0, 1.0)
    if "coverage" in metric_cols:
        bounds["coverage"] = (0.0, 100.0)
    if "coverage_robust" in metric_cols:
        bounds["coverage_robust"] = (0.0, 100.0)

    # Gate rates are percentages
    for col in metric_cols:
        if col.endswith("_rate"):
            bounds[col] = (0.0, 100.0)

    return bounds
