"""
Tidy data loader for experiment results.

Converts JSONL experiment outputs into a tidy DataFrame with one row per (run, policy) pair.
This is the single source of truth for all downstream analysis.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


DEFAULT_POLICIES = ("clone", "parallel_universe_prompt", "premium")


def load_results_jsonl(
    path: Path,
    policies: Tuple[str, ...] = DEFAULT_POLICIES,
    include_unhelpful: bool = False,  # Default to excluding unhelpful
) -> pd.DataFrame:
    """Load experiment results into tidy DataFrame.

    Each row represents one (run, policy) pair with all associated metrics.

    Args:
        path: Path to JSONL results file
        policies: Policies to include (default: well-behaved only)
        include_unhelpful: Whether to include the unhelpful policy

    Returns:
        Tidy DataFrame with columns:
        - Identifiers: run_id, seed, estimator
        - Regime: regime_n (sample size), regime_cov (oracle coverage)
        - Config flags: use_calib, use_covariates, rho, outer_cv, etc.
        - Policy: policy name
        - Point estimates: est, oracle_truth
        - Uncertainty: se, se_robust, ci_lo, ci_hi, ci_lo_robust, ci_hi_robust
        - Oracle info: n_oracle
        - Diagnostics: ess_rel, hill_alpha, a_bhat, mc_var_fraction
        - Gates: gate_overlap, gate_judge, gate_dr, gate_cap_stable, gate_refuse
        - Performance: runtime_s
    """
    if include_unhelpful:
        policies = policies + ("unhelpful",)

    rows = []
    with open(path) as f:
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

            # Extract base fields
            base_fields = {
                "run_id": r.get("run_id") or f"run_{line_num}",
                "seed": spec.get("seed_base", r.get("seed")),
                "regime_n": spec.get("sample_size"),
                "regime_cov": spec.get("oracle_coverage"),
                "estimator": spec.get("estimator"),
                "use_calib": bool(
                    extra.get(
                        "use_weight_calibration", r.get("use_weight_calibration", False)
                    )
                ),
                "use_covariates": bool(extra.get("use_covariates", False)),
                "rho": extra.get("rho", r.get("rho", 1.0)),
                "outer_cv": bool(extra.get("outer_cv", False)),
                "runtime_s": r.get("runtime_s"),
            }

            # Get metric dictionaries
            estimates = r.get("estimates", {})
            oracle_truths = r.get("oracle_truths", {})
            ses = r.get("standard_errors", {})
            ses_robust = r.get("robust_standard_errors", {})
            cis = r.get("confidence_intervals", {})
            cis_robust = r.get("robust_confidence_intervals", {})

            # Diagnostics
            ess_rel = r.get("ess_relative", {})
            hill_alpha = r.get("hill_alpha", {})
            a_bhat = r.get("bhattacharyya", {})

            # MC diagnostics (if using DR)
            mc_diag = r.get("mc_diagnostics", {})

            # Gates - compute from diagnostics if not stored
            diagnostics = r.get("diagnostics", {})

            # Oracle counts per policy (with fallback)
            n_oracle_per_policy = r.get("n_oracle_per_policy", {})
            n_oracle_global = r.get("n_oracle", 4989)

            for pol in policies:
                # Skip if no estimate for this policy
                if pol not in estimates:
                    continue

                # Extract CIs (and sort to ensure lo < hi)
                ci = cis.get(pol, [None, None])
                ci_robust = cis_robust.get(pol, [None, None])

                # Sanitize CI ordering
                if ci and ci[0] is not None and ci[1] is not None:
                    ci = sorted(ci)
                else:
                    ci = [None, None]

                if ci_robust and ci_robust[0] is not None and ci_robust[1] is not None:
                    ci_robust = sorted(ci_robust)
                else:
                    ci_robust = [None, None]

                # Compute gates from diagnostics if available
                pol_diag = diagnostics.get(pol, {})

                # Gate logic (simplified - customize based on your actual gates)
                gate_overlap = _check_overlap_gate(pol_diag, ess_rel.get(pol))
                gate_judge = _check_judge_gate(pol_diag)
                gate_dr = _check_dr_gate(pol_diag)
                gate_cap_stable = _check_cap_stability(pol_diag)
                gate_refuse = estimates.get(pol) is None or pd.isna(estimates.get(pol))

                # MC variance fraction
                mc_var_frac = None
                if pol in mc_diag:
                    mc_var_frac = mc_diag[pol].get("mc_var_fraction")

                row = {
                    **base_fields,
                    "policy": pol,
                    "est": estimates.get(pol),
                    "oracle_truth": oracle_truths.get(pol),
                    "se": ses.get(pol),
                    "se_robust": ses_robust.get(pol) if ses_robust else ses.get(pol),
                    "ci_lo": ci[0],
                    "ci_hi": ci[1],
                    "ci_lo_robust": ci_robust[0] if ci_robust[0] is not None else ci[0],
                    "ci_hi_robust": ci_robust[1] if ci_robust[1] is not None else ci[1],
                    "n_oracle": n_oracle_per_policy.get(pol, n_oracle_global),
                    "ess_rel": ess_rel.get(pol),
                    "hill_alpha": hill_alpha.get(pol),
                    "a_bhat": a_bhat.get(pol),
                    "mc_var_fraction": mc_var_frac,
                    "gate_overlap": gate_overlap,
                    "gate_judge": gate_judge,
                    "gate_dr": gate_dr,
                    "gate_cap_stable": gate_cap_stable,
                    "gate_refuse": gate_refuse,
                }
                rows.append(row)

    df = pd.DataFrame(rows)

    # Add derived columns
    if not df.empty:
        # Distinguish calibrated vs uncalibrated versions of estimators
        # For estimators that can be calibrated (dr-cpo), prepend "calibrated-" when use_calib=True
        # Also append "+cov" suffix when use_covariates=True
        calibratable_estimators = ["dr-cpo"]  # Add others as needed

        def adjust_estimator_name(row: pd.Series) -> str:
            est = row["estimator"]
            if est in calibratable_estimators and row.get("use_calib", False):
                est = f"calibrated-{est}"
            if row.get("use_covariates", False):
                est = f"{est}+cov"
            return str(est)

        df["estimator"] = df.apply(adjust_estimator_name, axis=1)

        # Also create a display name column that shows SNIPS instead of raw-ips
        def format_display_name(est: str) -> str:
            if est == "raw-ips":
                return "SNIPS"
            elif est == "raw-ips+cov":
                return "SNIPS+cov"
            return est

        df["estimator_display"] = df["estimator"].apply(format_display_name)
        # CI width
        df["ci_width"] = df["ci_hi"] - df["ci_lo"]
        df["ci_width_robust"] = df["ci_hi_robust"] - df["ci_lo_robust"]

        # Coverage indicators (fillna handles None values from failed CIs)
        df["covered"] = (
            (df["ci_lo"].fillna(float('inf')) <= df["oracle_truth"]) &
            (df["oracle_truth"] <= df["ci_hi"].fillna(float('-inf')))
        )
        df["covered_robust"] = (
            (df["ci_lo_robust"].fillna(float('inf')) <= df["oracle_truth"]) &
            (df["oracle_truth"] <= df["ci_hi_robust"].fillna(float('-inf')))
        )

        # Error metrics
        df["error"] = df["est"] - df["oracle_truth"]
        df["abs_error"] = df["error"].abs()
        df["squared_error"] = df["error"] ** 2

        # Oracle variance for debiasing
        df["oracle_var"] = df.apply(
            lambda r: _compute_oracle_variance(r["oracle_truth"], r["n_oracle"]), axis=1
        )
        df["debiased_squared_error"] = np.maximum(
            0, df["squared_error"] - df["oracle_var"]
        )

    return df


def _check_overlap_gate(diagnostics: Dict[str, Any], ess_rel: Optional[float]) -> bool:
    """Check if overlap gate passes.

    Gate passes if ESS > 10% or Hill alpha > 2.
    """
    if ess_rel is not None and ess_rel < 0.1:
        return False
    hill = diagnostics.get("hill_alpha")
    if hill is not None and hill < 2.0:
        return False
    return True


def _check_judge_gate(diagnostics: Dict[str, Any]) -> bool:
    """Check if judge gate passes.

    Gate passes if judge reliability metrics are acceptable.
    """
    # Placeholder - customize based on your actual judge diagnostics
    kendall_tau = diagnostics.get("kendall_tau")
    if kendall_tau is not None and kendall_tau < 0.3:
        return False
    return True


def _check_dr_gate(diagnostics: Dict[str, Any]) -> bool:
    """Check if DR gate passes.

    Gate passes if DR orthogonality conditions are met.
    """
    # Check if orthogonality score CI contains 0
    orth_ci = diagnostics.get("orthogonality_ci", [None, None])
    if orth_ci[0] is not None and orth_ci[1] is not None:
        return bool(orth_ci[0] <= 0 <= orth_ci[1])
    return True


def _check_cap_stability(diagnostics: Dict[str, Any]) -> bool:
    """Check if variance cap is stable.

    Gate passes if cap activity is reasonable (not too steep).
    """
    cap_activity = diagnostics.get("cap_activity_rate", 0)
    if cap_activity > 0.5:  # More than 50% of weights capped
        return False
    return True


def _compute_oracle_variance(oracle_truth: float, n_oracle: int) -> float:
    """Compute conservative Bernoulli variance for oracle.

    Args:
        oracle_truth: Oracle point estimate (probability)
        n_oracle: Number of samples used for oracle

    Returns:
        Variance of oracle estimate
    """
    if pd.isna(oracle_truth) or pd.isna(n_oracle) or n_oracle <= 0:
        return float(np.nan)
    # Conservative Bernoulli variance
    return float(min(oracle_truth * (1 - oracle_truth), 0.25) / n_oracle)


def deduplicate_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate runs, keeping first occurrence.

    Args:
        df: Tidy DataFrame from load_results_jsonl

    Returns:
        DataFrame with duplicates removed
    """
    # Define unique run key
    key_cols = ["estimator", "regime_n", "regime_cov", "seed", "use_calib", "policy"]

    # Keep first occurrence of each unique combination
    return df.drop_duplicates(subset=key_cols, keep="first")


def filter_by_regime(
    df: pd.DataFrame,
    sample_sizes: Optional[List[int]] = None,
    coverages: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Filter DataFrame to specific regimes.

    Args:
        df: Tidy DataFrame
        sample_sizes: List of sample sizes to include (None = all)
        coverages: List of oracle coverages to include (None = all)

    Returns:
        Filtered DataFrame
    """
    if sample_sizes is not None:
        df = df[df["regime_n"].isin(sample_sizes)]
    if coverages is not None:
        df = df[df["regime_cov"].isin(coverages)]
    return df


def filter_contaminated_covariate_runs(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Filter out runs contaminated by covariate computation bug.

    Bug context: Prior to Oct 2024, calibration used word count but fresh draws
    used log10(char count), causing systematic +0.06 bias when use_covariates=True.

    This function removes all runs with use_covariates=True to show only valid results.

    Args:
        df: Tidy DataFrame from load_results_jsonl
        verbose: Print filtering statistics

    Returns:
        DataFrame with contaminated runs removed
    """
    if "use_covariates" not in df.columns:
        if verbose:
            print("Warning: 'use_covariates' column not found - no filtering applied")
        return df

    n_before = len(df)
    n_contaminated = df["use_covariates"].sum()

    # Filter out contaminated runs
    df_clean = df[~df["use_covariates"]].copy()

    if verbose and n_contaminated > 0:
        print(f"Filtered out {n_contaminated}/{n_before} contaminated rows (use_covariates=True)")
        print(f"Remaining: {len(df_clean)} clean rows")

    return df_clean
