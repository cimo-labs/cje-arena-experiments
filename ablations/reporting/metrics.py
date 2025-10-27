"""
Metric registry with pure functions over tidy DataFrames.

Each function takes a tidy DataFrame and grouping columns, returns aggregated metrics.
All metrics include numerical guards and consistent handling of edge cases.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from scipy import stats

# Small epsilon for numerical stability
EPS = 1e-12


def rmse_debiased(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Compute oracle-noise-debiased RMSE (RMSE^d) per group.

    RMSE^d = sqrt(mean(max(0, (est - oracle)^2 - var_oracle)))

    Args:
        df: Tidy DataFrame with columns: est, oracle_truth, oracle_var
        by: Grouping columns

    Returns:
        DataFrame with grouping columns + rmse_d
    """

    def compute_rmse_d(group: pd.DataFrame) -> float:
        # Use precomputed debiased squared errors
        debiased_sq = group["debiased_squared_error"].dropna()
        if len(debiased_sq) == 0:
            return float(np.nan)
        return float(np.sqrt(np.mean(debiased_sq)))

    result = (
        df.groupby(by, group_keys=False)
        .apply(compute_rmse_d)
        .reset_index(name="rmse_d")
    )
    return result


def mae(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Compute Mean Absolute Error per group.

    Args:
        df: Tidy DataFrame with columns: abs_error
        by: Grouping columns

    Returns:
        DataFrame with grouping columns + mae
    """
    result = df.groupby(by)["abs_error"].mean().reset_index(name="mae")
    return result


def interval_score_oa(
    df: pd.DataFrame, by: List[str], alpha: float = 0.05
) -> pd.DataFrame:
    """Compute oracle-adjusted interval score per group.

    IS = width + (2/α) * coverage_penalty
    Uses geometric mean across policies.

    Args:
        df: Tidy DataFrame with CI and oracle columns
        by: Grouping columns
        alpha: Significance level (0.05 for 95% CI)

    Returns:
        DataFrame with grouping columns + interval_score_oa
    """

    def compute_is(group: pd.DataFrame) -> float:
        scores = []
        for _, row in group.iterrows():
            if pd.isna(row["ci_lo_robust"]) or pd.isna(row["ci_hi_robust"]):
                continue
            if pd.isna(row["oracle_truth"]):
                continue

            width = row["ci_hi_robust"] - row["ci_lo_robust"]
            truth = row["oracle_truth"]

            # Coverage penalty
            if truth < row["ci_lo_robust"]:
                penalty = (2 / alpha) * (row["ci_lo_robust"] - truth)
            elif truth > row["ci_hi_robust"]:
                penalty = (2 / alpha) * (truth - row["ci_hi_robust"])
            else:
                penalty = 0

            score = width + penalty
            if score > 0:  # Guard against non-positive values
                scores.append(score)

        if not scores:
            return float(np.nan)

        # Geometric mean with epsilon guard
        return float(np.exp(np.mean(np.log(np.array(scores) + EPS))))

    result = (
        df.groupby(by, group_keys=False)
        .apply(compute_is)
        .reset_index(name="interval_score_oa")
    )
    return result


def calib_score(df: pd.DataFrame, by: List[str], target: float = 0.95) -> pd.DataFrame:
    """Compute calibration score: (coverage - target).

    Positive values = over-coverage (conservative), negative = under-coverage.

    Args:
        df: Tidy DataFrame with coverage indicator
        by: Grouping columns
        target: Target coverage level

    Returns:
        DataFrame with grouping columns + calib_score (as percentage, signed)
    """

    def compute_calib(group: pd.DataFrame) -> float:
        covered = group["covered_robust"].dropna()
        if len(covered) == 0:
            return float(np.nan)
        coverage = covered.mean()
        return float((coverage - target) * 100)  # Convert to percentage, signed

    result = (
        df.groupby(by, group_keys=False)
        .apply(compute_calib)
        .reset_index(name="calib_score")
    )
    return result


def se_geomean(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Compute geometric mean of standard errors (sharpness proxy).

    Args:
        df: Tidy DataFrame with se_robust column
        by: Grouping columns

    Returns:
        DataFrame with grouping columns + se_geomean
    """

    def compute_se_gm(group: pd.DataFrame) -> float:
        ses = group["se_robust"].dropna()
        ses = ses[ses > 0]  # Filter out non-positive values
        if len(ses) == 0:
            return float(np.nan)
        # Geometric mean with epsilon guard
        return float(np.exp(np.mean(np.log(ses + EPS))))

    result = (
        df.groupby(by, group_keys=False)
        .apply(compute_se_gm)
        .reset_index(name="se_geomean")
    )
    return result


def ranking_metrics(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Compute ranking metrics: Kendall tau, pairwise accuracy, top-1 accuracy/regret.

    Args:
        df: Tidy DataFrame with est and oracle_truth
        by: Grouping columns (should include estimator/seed but not policy)

    Returns:
        DataFrame with grouping columns + ranking metrics
    """
    from scipy.stats import kendalltau
    from itertools import combinations

    def compute_ranking(group: pd.DataFrame) -> pd.Series:
        # Get estimates and truths for each policy
        policies = group["policy"].unique()
        if len(policies) < 2:
            return pd.Series(
                {
                    "kendall_tau": np.nan,
                    "pairwise_acc": np.nan,
                    "top1_acc": np.nan,
                    "top1_regret": np.nan,
                }
            )

        est_dict = {}
        truth_dict = {}
        for pol in policies:
            pol_data = group[group["policy"] == pol]
            if not pol_data.empty:
                # Take mean if multiple values (shouldn't happen at experiment level)
                est_dict[pol] = pol_data["est"].mean()
                truth_dict[pol] = pol_data["oracle_truth"].iloc[0]

        # Filter to valid pairs
        valid_policies = [
            p
            for p in policies
            if p in est_dict
            and p in truth_dict
            and not pd.isna(est_dict[p])
            and not pd.isna(truth_dict[p])
        ]

        if len(valid_policies) < 2:
            return pd.Series(
                {
                    "kendall_tau": np.nan,
                    "pairwise_acc": np.nan,
                    "top1_acc": np.nan,
                    "top1_regret": np.nan,
                }
            )

        est_values = [est_dict[p] for p in valid_policies]
        truth_values = [truth_dict[p] for p in valid_policies]

        # Kendall tau
        tau, _ = kendalltau(est_values, truth_values)

        # Pairwise accuracy
        correct_pairs = 0
        total_pairs = 0
        for i, j in combinations(range(len(valid_policies)), 2):
            est_order = est_values[i] > est_values[j]
            truth_order = truth_values[i] > truth_values[j]
            if est_order == truth_order:
                correct_pairs += 1
            total_pairs += 1
        pairwise = correct_pairs / total_pairs if total_pairs > 0 else np.nan

        # Top-1 accuracy and regret
        if len(valid_policies) >= 3:  # Need at least 3 for meaningful top-1
            best_est_idx = np.argmax(est_values)
            best_truth_idx = np.argmax(truth_values)
            top1_correct = best_est_idx == best_truth_idx

            # Regret: difference from true best
            selected_truth = truth_values[best_est_idx]
            best_truth = truth_values[best_truth_idx]
            regret = best_truth - selected_truth
        else:
            top1_correct = np.nan
            regret = np.nan

        return pd.Series(
            {
                "kendall_tau": tau,
                "pairwise_acc": pairwise,
                "top1_acc": (
                    float(top1_correct) if not pd.isna(top1_correct) else np.nan
                ),
                "top1_regret": regret,
            }
        )

    # First compute at experiment level, then aggregate
    # Experiment level: estimator + seed + regime
    experiment_cols = list(set(by) | {"seed", "regime_n", "regime_cov"})
    experiment_cols = [c for c in experiment_cols if c in df.columns and c != "policy"]

    # Compute per-experiment rankings
    per_exp = (
        df.groupby(experiment_cols, group_keys=False)
        .apply(compute_ranking)
        .reset_index()
    )

    # Now aggregate to requested level
    if set(by) == set(experiment_cols):
        # Already at experiment level
        return per_exp
    else:
        # Aggregate from experiment level to requested level
        agg_dict = {
            "kendall_tau": "mean",
            "pairwise_acc": "mean",
            "top1_acc": "mean",
            "top1_regret": "mean",
        }
        result = per_exp.groupby(by).agg(agg_dict).reset_index()
        return result


def runtime_stats(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Compute runtime statistics per group (deduped by run_id).

    Args:
        df: Tidy DataFrame with runtime_s column
        by: Grouping columns

    Returns:
        DataFrame with grouping columns + runtime_median, runtime_mean
    """
    # Deduplicate at run level to avoid counting same run multiple times
    cols = list(dict.fromkeys(["run_id"] + by + ["runtime_s"]))
    cols = [c for c in cols if c in df.columns]  # Filter to existing columns

    if "run_id" in df.columns:
        run_level = df[cols].drop_duplicates(subset=["run_id"])
    else:
        # Fallback if no run_id
        run_level = df[cols].drop_duplicates()

    result = (
        run_level.groupby(by, dropna=False)["runtime_s"]
        .agg(runtime_median="median", runtime_mean="mean", runtime_std="std")
        .reset_index()
    )
    return result


def gate_pass_rates(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Compute gate pass rates per group.

    Args:
        df: Tidy DataFrame with gate columns
        by: Grouping columns

    Returns:
        DataFrame with grouping columns + gate pass rates (as percentages)
    """
    gate_cols = [
        "gate_overlap",
        "gate_judge",
        "gate_dr",
        "gate_cap_stable",
        "gate_refuse",
    ]

    def compute_gates(group: pd.DataFrame) -> Dict[str, float]:
        results = {}
        for gate in gate_cols:
            if gate in group.columns:
                # Convert to pass rate (True = pass, False = fail)
                if gate == "gate_refuse":
                    # Refuse is inverted (True = bad, want low rate)
                    results[f"{gate}_rate"] = (group[gate] == True).mean() * 100
                else:
                    # Normal gates (True = good, want high rate)
                    results[f"{gate}_rate"] = (group[gate] == True).mean() * 100
            else:
                results[f"{gate}_rate"] = np.nan
        return pd.Series(results)  # type: ignore

    result = df.groupby(by, group_keys=False).apply(compute_gates).reset_index()
    return result


def coverage_stats(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Compute coverage statistics per group.

    Args:
        df: Tidy DataFrame with coverage columns
        by: Grouping columns

    Returns:
        DataFrame with coverage rates and CI widths
    """
    result = (
        df.groupby(by)
        .agg(
            coverage=("covered", "mean"),
            coverage_robust=("covered_robust", "mean"),
            ci_width_mean=("ci_width", "mean"),
            ci_width_robust_mean=("ci_width_robust", "mean"),
            ci_width_median=("ci_width", "median"),
            ci_width_robust_median=("ci_width_robust", "median"),
        )
        .reset_index()
    )

    # Convert coverage to percentage
    result["coverage"] *= 100
    result["coverage_robust"] *= 100

    return result


def overlap_diagnostics(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Compute overlap diagnostics per group.

    Args:
        df: Tidy DataFrame with ESS, Hill alpha, Bhattacharyya columns
        by: Grouping columns

    Returns:
        DataFrame with overlap metrics
    """
    result = (
        df.groupby(by)
        .agg(
            ess_rel_mean=("ess_rel", "mean"),
            ess_rel_median=("ess_rel", "median"),
            ess_rel_min=("ess_rel", "min"),
            hill_alpha_mean=("hill_alpha", "mean"),
            hill_alpha_median=("hill_alpha", "median"),
            hill_alpha_min=("hill_alpha", "min"),
            a_bhat_mean=("a_bhat", "mean"),
            a_bhat_median=("a_bhat", "median"),
        )
        .reset_index()
    )

    # Convert ESS to percentage
    for col in ["ess_rel_mean", "ess_rel_median", "ess_rel_min"]:
        if col in result.columns:
            result[col] *= 100

    return result


def mc_variance_diagnostics(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Compute Monte Carlo variance diagnostics for DR estimators.

    Args:
        df: Tidy DataFrame with mc_var_fraction column
        by: Grouping columns

    Returns:
        DataFrame with MC variance statistics
    """
    # Check if MC diagnostics exist
    if "mc_var_fraction" not in df.columns:
        # Return empty frame with expected columns
        result = pd.DataFrame(
            columns=by
            + ["mc_var_fraction_mean", "mc_var_fraction_median", "mc_var_fraction_max"]
        )
        return result

    # Only include rows with MC diagnostics
    df_mc = df[df["mc_var_fraction"].notna()]

    if df_mc.empty:
        # Return empty frame with expected columns
        result = pd.DataFrame(
            columns=by
            + ["mc_var_fraction_mean", "mc_var_fraction_median", "mc_var_fraction_max"]
        )
        return result

    result = (
        df_mc.groupby(by)
        .agg(
            mc_var_fraction_mean=("mc_var_fraction", "mean"),
            mc_var_fraction_median=("mc_var_fraction", "median"),
            mc_var_fraction_max=("mc_var_fraction", "max"),
        )
        .reset_index()
    )

    # Convert to percentage
    for col in [
        "mc_var_fraction_mean",
        "mc_var_fraction_median",
        "mc_var_fraction_max",
    ]:
        if col in result.columns:
            result[col] *= 100

    return result


def compute_all_metrics(
    df: pd.DataFrame, by: List[str], include_debiased: bool = True
) -> pd.DataFrame:
    """Compute all metrics and merge into single DataFrame.

    Args:
        df: Tidy DataFrame
        by: Grouping columns
        include_debiased: Whether to include debiased metrics

    Returns:
        DataFrame with all metrics merged on grouping columns
    """
    # For accuracy metrics, exclude unhelpful policy
    # (RMSE, MAE, IS^OA, calibration, SE, coverage should measure accuracy on real policies)
    df_no_unhelpful = (
        df[df["policy"] != "unhelpful"] if "unhelpful" in df["policy"].values else df
    )

    # Start with base metrics (computed WITHOUT unhelpful)
    metrics = [
        rmse_debiased(df_no_unhelpful, by),
        mae(df_no_unhelpful, by),
        interval_score_oa(df_no_unhelpful, by),
        calib_score(df_no_unhelpful, by),
        se_geomean(df_no_unhelpful, by),
        coverage_stats(df_no_unhelpful, by),
        runtime_stats(df_no_unhelpful, by),
        gate_pass_rates(df_no_unhelpful, by),
        overlap_diagnostics(df_no_unhelpful, by),
    ]

    # Add ranking metrics if policy is not in grouping
    # (computed WITH unhelpful to properly test ranking ability)
    if "policy" not in by:
        metrics.append(ranking_metrics(df, by))  # Use full df with unhelpful

    # Add MC diagnostics for DR estimators (also exclude unhelpful)
    dr_mask = (
        df_no_unhelpful["estimator"]
        .astype(str)
        .str.contains(r"\b(?:dr|tmle|mrdr)\b", case=False, regex=True, na=False)
    )
    dr_estimators = df_no_unhelpful[dr_mask]
    if not dr_estimators.empty:
        metrics.append(mc_variance_diagnostics(dr_estimators, by))

    # Merge all metrics
    result = metrics[0]
    for metric_df in metrics[1:]:
        if not metric_df.empty:
            result = result.merge(metric_df, on=by, how="outer")

    return result
