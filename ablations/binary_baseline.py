#!/usr/bin/env python3
"""
Binary baseline estimators for MBPP ablation study.

Implements Lee et al. misclassification adjustment (Rogan-Gladen estimator)
for comparison against continuous CJE calibration.

Key estimators:
- binary_naive: Raw binary judge predictions (Ẑ mean)
- binary_adjusted: Rogan-Gladen misclassification adjustment

Key audits:
- Youden's J validity: Se + Sp > 1 required for valid adjustment
- Confusion matrix transport: P(Ẑ|Y) stability across policies
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ConfusionStats:
    """Statistics from a confusion matrix."""
    sensitivity: float  # Se = P(Ẑ=1|Y=1) = true positive rate
    specificity: float  # Sp = P(Ẑ=0|Y=0) = true negative rate
    n_pos: int  # Number of positive oracle labels
    n_neg: int  # Number of negative oracle labels
    youdens_j: float  # Se + Sp - 1


@dataclass
class BinaryEstimate:
    """Result of a binary estimation."""
    estimate: float  # Point estimate of pass rate
    variance: Optional[float]  # Variance estimate (if available)
    se: Optional[float]  # Standard error
    valid: bool  # Whether estimate is valid
    method: str  # 'naive' or 'adjusted'
    diagnostics: Dict  # Additional diagnostic info


def estimate_confusion_stats(
    y_oracle: np.ndarray,
    z_binary: np.ndarray,
) -> ConfusionStats:
    """Estimate sensitivity and specificity from labeled calibration set.

    Sensitivity (Se) = P(Ẑ=1 | Y=1) — probability judge says YES given correct code
    Specificity (Sp) = P(Ẑ=0 | Y=0) — probability judge says NO given incorrect code

    Args:
        y_oracle: Oracle labels (0/1 for fail/pass)
        z_binary: Binary judge predictions (0/1 for NO/YES)

    Returns:
        ConfusionStats with Se, Sp, sample sizes, and Youden's J
    """
    y_oracle = np.asarray(y_oracle)
    z_binary = np.asarray(z_binary)

    # Identify positive and negative oracle labels
    pos_mask = (y_oracle == 1)
    neg_mask = (y_oracle == 0)

    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    # Sensitivity: P(Ẑ=1 | Y=1)
    if n_pos > 0:
        se = z_binary[pos_mask].mean()
    else:
        se = np.nan

    # Specificity: P(Ẑ=0 | Y=0)
    if n_neg > 0:
        sp = (1 - z_binary[neg_mask]).mean()
    else:
        sp = np.nan

    # Youden's J index
    if not np.isnan(se) and not np.isnan(sp):
        youdens_j = se + sp - 1.0
    else:
        youdens_j = np.nan

    return ConfusionStats(
        sensitivity=se,
        specificity=sp,
        n_pos=int(n_pos),
        n_neg=int(n_neg),
        youdens_j=youdens_j,
    )


def rogan_gladen_estimate(
    p_hat: float,
    se: float,
    sp: float,
    j_threshold: float = 0.05,
) -> Tuple[float, bool]:
    """Compute Rogan-Gladen misclassification-adjusted prevalence.

    The adjustment formula:
        θ̂ = (p̂ + Sp - 1) / (Se + Sp - 1)

    where:
        p̂ = observed proportion of positive predictions
        Se = sensitivity = P(Ẑ=1|Y=1)
        Sp = specificity = P(Ẑ=0|Y=0)

    Args:
        p_hat: Observed proportion of YES predictions
        se: Estimated sensitivity
        sp: Estimated specificity
        j_threshold: Minimum Youden's J for valid adjustment (default 0.05)

    Returns:
        Tuple of (adjusted_estimate, is_valid)
        - If Youden's J <= j_threshold, returns (np.nan, False)
    """
    youdens_j = se + sp - 1.0

    # Critical validity check: J must be positive
    # If J ≈ 0, binary judge is near-random and cannot be calibrated
    if np.isnan(youdens_j) or youdens_j <= j_threshold:
        return np.nan, False

    theta = (p_hat + sp - 1.0) / youdens_j

    # Clip to [0, 1] — values outside indicate model misspecification
    theta_clipped = np.clip(theta, 0.0, 1.0)

    return theta_clipped, True


def rogan_gladen_variance(
    p_hat: float,
    se: float,
    sp: float,
    n_test: int,
    n_cal_pos: int,
    n_cal_neg: int,
) -> float:
    """Variance estimate for Rogan-Gladen estimator.

    Accounts for uncertainty in both:
    1. Test set proportion (p̂)
    2. Calibration set estimates (Se, Sp)

    Based on Lang-Reiczigel variance formula, adapted from Lee et al.

    Args:
        p_hat: Observed proportion of YES predictions
        se: Estimated sensitivity
        sp: Estimated specificity
        n_test: Number of test samples
        n_cal_pos: Number of positive calibration samples
        n_cal_neg: Number of negative calibration samples

    Returns:
        Variance of the Rogan-Gladen estimator
    """
    j = se + sp - 1.0

    if np.isnan(j) or j <= 0.05:
        return np.nan

    # Variance components
    var_p = p_hat * (1 - p_hat) / n_test

    var_se = se * (1 - se) / n_cal_pos if n_cal_pos > 0 else 0
    var_sp = sp * (1 - sp) / n_cal_neg if n_cal_neg > 0 else 0

    # Delta method: Var(θ̂) ≈ (∂θ/∂p)² Var(p) + (∂θ/∂se)² Var(se) + (∂θ/∂sp)² Var(sp)
    # ∂θ/∂p = 1/J
    # ∂θ/∂se = -θ/J (via quotient rule)
    # ∂θ/∂sp = (1 - θ)/J (via quotient rule)

    theta = (p_hat + sp - 1.0) / j

    var_theta = (
        (1/j)**2 * var_p +
        (theta/j)**2 * var_se +
        ((1 - theta)/j)**2 * var_sp
    )

    return var_theta


def lee_et_al_confidence_interval(
    p: float,
    q0: float,
    q1: float,
    n: int,
    m0: int,
    m1: int,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Compute Lee et al. (2026) confidence interval for Rogan-Gladen estimator.

    This implements Eq. 6 from "How to Correctly Report LLM-as-a-Judge Evaluations"
    with Agresti-Coull adjustment and bias correction term.

    Args:
        p: Observed positive rate in test set
        q0: Specificity (true negative rate) from calibration set
        q1: Sensitivity (true positive rate) from calibration set
        n: Test set size
        m0: Number of negatives in calibration set
        m1: Number of positives in calibration set
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound) for 1-alpha confidence interval
    """
    z = stats.norm.ppf(1 - alpha / 2)

    # Agresti-Coull adjustment
    p_adj = (n * p + z**2 / 2) / (n + z**2)
    q0_adj = (m0 * q0 + 1) / (m0 + 2)
    q1_adj = (m1 * q1 + 1) / (m1 + 2)
    n_adj = n + z**2
    m0_adj = m0 + 2
    m1_adj = m1 + 2

    # Check validity (Youden's J must be positive)
    j_adj = q0_adj + q1_adj - 1
    if j_adj <= 0:
        return np.nan, np.nan

    # Adjusted estimate
    theta = (p_adj + q0_adj - 1) / j_adj

    # Bias correction term (Eq. 6)
    dth = 2 * z**2 * (-(1 - theta) * q0_adj * (1 - q0_adj) / m0_adj
                      + theta * q1_adj * (1 - q1_adj) / m1_adj)

    # Standard error (accounts for both test and calibration uncertainty)
    var_p = p_adj * (1 - p_adj) / n_adj
    var_q0 = (1 - theta)**2 * q0_adj * (1 - q0_adj) / m0_adj
    var_q1 = theta**2 * q1_adj * (1 - q1_adj) / m1_adj
    se = np.sqrt(var_p + var_q0 + var_q1) / j_adj

    # Confidence interval with bias correction
    lower = theta + dth - z * se
    upper = theta + dth + z * se

    # Clip to [0, 1]
    lower = max(0.0, min(1.0, lower))
    upper = max(0.0, min(1.0, upper))

    return lower, upper


def binary_naive_estimate(
    z_binary: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None,
) -> BinaryEstimate:
    """Naive binary estimator: raw proportion of YES predictions.

    Args:
        z_binary: Binary judge predictions (0/1)
        cluster_ids: Optional cluster IDs for clustered SE

    Returns:
        BinaryEstimate with point estimate and SE
    """
    z_binary = np.asarray(z_binary)
    n = len(z_binary)

    # Point estimate
    p_hat = z_binary.mean()

    # Standard error
    if cluster_ids is not None:
        # Clustered SE
        cluster_ids = np.asarray(cluster_ids)
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        # Cluster means
        cluster_means = np.array([
            z_binary[cluster_ids == c].mean()
            for c in unique_clusters
        ])

        # Clustered variance
        cluster_var = cluster_means.var(ddof=1)
        se = np.sqrt(cluster_var / n_clusters)
    else:
        # Simple SE
        var_p = p_hat * (1 - p_hat) / n
        se = np.sqrt(var_p)

    return BinaryEstimate(
        estimate=p_hat,
        variance=se**2,
        se=se,
        valid=True,
        method='naive',
        diagnostics={'n': n},
    )


def binary_adjusted_estimate(
    z_binary: np.ndarray,
    confusion_stats: ConfusionStats,
    cluster_ids: Optional[np.ndarray] = None,
    j_threshold: float = 0.05,
) -> BinaryEstimate:
    """Rogan-Gladen adjusted binary estimator.

    Args:
        z_binary: Binary judge predictions (0/1) for test set
        confusion_stats: Pre-computed confusion stats from calibration set
        cluster_ids: Optional cluster IDs for clustered SE
        j_threshold: Minimum Youden's J for valid adjustment

    Returns:
        BinaryEstimate with adjusted point estimate and SE
    """
    z_binary = np.asarray(z_binary)
    n = len(z_binary)

    # Get raw proportion
    p_hat = z_binary.mean()

    # Check validity
    se = confusion_stats.sensitivity
    sp = confusion_stats.specificity
    j = confusion_stats.youdens_j

    if np.isnan(j) or j <= j_threshold:
        return BinaryEstimate(
            estimate=np.nan,
            variance=np.nan,
            se=np.nan,
            valid=False,
            method='adjusted',
            diagnostics={
                'youdens_j': j,
                'p_hat': p_hat,
                'reason': f'Youden J ({j:.3f}) <= threshold ({j_threshold})',
            },
        )

    # Compute adjusted estimate
    theta, valid = rogan_gladen_estimate(p_hat, se, sp, j_threshold)

    if not valid:
        return BinaryEstimate(
            estimate=np.nan,
            variance=np.nan,
            se=np.nan,
            valid=False,
            method='adjusted',
            diagnostics={'youdens_j': j, 'p_hat': p_hat},
        )

    # Compute variance
    var_theta = rogan_gladen_variance(
        p_hat, se, sp, n,
        confusion_stats.n_pos, confusion_stats.n_neg
    )
    se_theta = np.sqrt(var_theta) if not np.isnan(var_theta) else np.nan

    return BinaryEstimate(
        estimate=theta,
        variance=var_theta,
        se=se_theta,
        valid=True,
        method='adjusted',
        diagnostics={
            'youdens_j': j,
            'p_hat': p_hat,
            'sensitivity': se,
            'specificity': sp,
            'n_cal_pos': confusion_stats.n_pos,
            'n_cal_neg': confusion_stats.n_neg,
        },
    )


def confusion_matrix_transport_audit(
    stats_base: ConfusionStats,
    stats_target: ConfusionStats,
    alpha: float = 0.05,
) -> Dict:
    """Test if confusion matrix (Se, Sp) is stable from base to target policy.

    Uses two-proportion z-tests for Se and Sp separately,
    with Bonferroni correction for multiple testing.

    This is the binary analog of the mean-residual transport audit for
    continuous scores. If P(Ẑ|Y) differs between policies, the calibrator
    (confusion matrix) doesn't transfer and adjusted estimates are biased.

    Args:
        stats_base: Confusion stats from base policy calibration set
        stats_target: Confusion stats from target policy (with oracle labels)
        alpha: Significance level for audit

    Returns:
        Dictionary with:
        - passes_audit: Whether confusion matrix appears stable
        - p_se: p-value for sensitivity comparison
        - p_sp: p-value for specificity comparison
        - p_combined: Bonferroni-corrected combined p-value
        - delta_se: Difference in sensitivity
        - delta_sp: Difference in specificity
    """
    # Test for sensitivity equality
    se_base, se_target = stats_base.sensitivity, stats_target.sensitivity
    n_pos_base, n_pos_target = stats_base.n_pos, stats_target.n_pos

    if n_pos_base > 0 and n_pos_target > 0 and not np.isnan(se_base) and not np.isnan(se_target):
        # Pooled estimate
        se_pooled = (se_base * n_pos_base + se_target * n_pos_target) / (n_pos_base + n_pos_target)
        se_diff = se_base - se_target
        se_se = np.sqrt(se_pooled * (1 - se_pooled) * (1/n_pos_base + 1/n_pos_target))

        if se_se > 0:
            z_se = se_diff / se_se
            p_se = 2 * (1 - stats.norm.cdf(abs(z_se)))
        else:
            p_se = 1.0
    else:
        p_se = 1.0
        se_diff = np.nan

    # Test for specificity equality
    sp_base, sp_target = stats_base.specificity, stats_target.specificity
    n_neg_base, n_neg_target = stats_base.n_neg, stats_target.n_neg

    if n_neg_base > 0 and n_neg_target > 0 and not np.isnan(sp_base) and not np.isnan(sp_target):
        sp_pooled = (sp_base * n_neg_base + sp_target * n_neg_target) / (n_neg_base + n_neg_target)
        sp_diff = sp_base - sp_target
        sp_se = np.sqrt(sp_pooled * (1 - sp_pooled) * (1/n_neg_base + 1/n_neg_target))

        if sp_se > 0:
            z_sp = sp_diff / sp_se
            p_sp = 2 * (1 - stats.norm.cdf(abs(z_sp)))
        else:
            p_sp = 1.0
    else:
        p_sp = 1.0
        sp_diff = np.nan

    # Bonferroni correction
    p_combined = min(1.0, 2 * min(p_se, p_sp))
    passes_audit = p_combined >= alpha

    return {
        'passes_audit': passes_audit,
        'p_se': p_se,
        'p_sp': p_sp,
        'p_combined': p_combined,
        'delta_se': se_diff if not np.isnan(se_diff) else stats_target.sensitivity - stats_base.sensitivity,
        'delta_sp': sp_diff if not np.isnan(sp_diff) else stats_target.specificity - stats_base.specificity,
        'se_base': stats_base.sensitivity,
        'se_target': stats_target.sensitivity,
        'sp_base': stats_base.specificity,
        'sp_target': stats_target.specificity,
    }


def check_minimum_sample_sizes(
    n_pos: int,
    n_neg: int,
    min_samples: int = 10,
) -> Tuple[bool, str]:
    """Check if calibration set has sufficient samples for stable estimates.

    Args:
        n_pos: Number of positive oracle labels in calibration set
        n_neg: Number of negative oracle labels in calibration set
        min_samples: Minimum required per class

    Returns:
        Tuple of (is_valid, reason)
    """
    if n_pos < min_samples:
        return False, f"Insufficient positive samples ({n_pos} < {min_samples})"
    if n_neg < min_samples:
        return False, f"Insufficient negative samples ({n_neg} < {min_samples})"
    return True, "OK"


# Convenience function for running complete binary analysis
def run_binary_analysis(
    y_cal: np.ndarray,
    z_cal: np.ndarray,
    z_test: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None,
) -> Dict:
    """Run complete binary analysis: naive and adjusted estimates with diagnostics.

    Args:
        y_cal: Oracle labels for calibration set
        z_cal: Binary judge predictions for calibration set
        z_test: Binary judge predictions for test set
        cluster_ids: Optional cluster IDs for test set

    Returns:
        Dictionary with estimates and diagnostics
    """
    # Get confusion stats from calibration set
    confusion_stats = estimate_confusion_stats(y_cal, z_cal)

    # Check sample sizes
    sample_valid, sample_reason = check_minimum_sample_sizes(
        confusion_stats.n_pos, confusion_stats.n_neg
    )

    # Naive estimate
    naive_est = binary_naive_estimate(z_test, cluster_ids)

    # Adjusted estimate
    adjusted_est = binary_adjusted_estimate(z_test, confusion_stats, cluster_ids)

    return {
        'naive': naive_est,
        'adjusted': adjusted_est,
        'confusion_stats': confusion_stats,
        'sample_valid': sample_valid,
        'sample_reason': sample_reason,
    }


def get_thresholded_binary(samples: List[Dict], threshold: float = 0.5) -> np.ndarray:
    """Convert continuous judge_score to binary predictions via thresholding.

    This allows testing how binary calibration (Rogan-Gladen) performs when using
    thresholded continuous scores instead of native A/B binary predictions.

    Args:
        samples: List of sample dicts with metadata.judge_score (0-1 scale)
        threshold: Score threshold on 0-1 scale

    Returns:
        Binary predictions (1 if score >= threshold, 0 otherwise)
        NaN for samples with missing judge_score
    """
    binary = []
    for s in samples:
        score = s.get('metadata', {}).get('judge_score')
        if score is None:
            binary.append(np.nan)
        else:
            # Score is already on 0-1 scale
            binary.append(1 if score >= threshold else 0)
    return np.array(binary)
