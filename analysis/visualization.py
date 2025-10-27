"""Visualization generation for CJE analysis.

This module handles generating all visualization plots and dashboards.

Following CLAUDE.md: Do one thing well - this module only handles visualization.
"""

import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional
import warnings

# Check if visualization is available
try:
    import matplotlib.pyplot as plt
    from cje.visualization import (
        plot_weight_dashboard_summary,
        plot_weight_dashboard_detailed,
        plot_calibration_comparison,
        plot_policy_estimates,
        plot_dr_dashboard,
    )

    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False
    warnings.warn(
        "Visualization dependencies not available. Install with: pip install cje[viz]"
    )

from .results import load_oracle_ground_truth


def generate_visualizations(
    results: Any,
    dataset: Any,
    calibrated_dataset: Any,
    estimator: Any,
    sampler: Any,
    args: Any,
    summary_data: Dict[str, Any],
    cal_result: Optional[Any] = None,
) -> None:
    """Generate all visualization plots.

    Args:
        results: EstimationResult object
        dataset: Original dataset (for oracle comparison)
        calibrated_dataset: Dataset with calibrated rewards
        estimator: Fitted estimator
        sampler: PrecomputedSampler
        args: Command-line arguments
        summary_data: Summary statistics from results
        cal_result: Optional calibration result
    """
    if not VIZ_AVAILABLE or args.no_plots:
        return

    plot_dir = (
        Path(args.plot_dir) if args.plot_dir else Path(args.data).parent / "plots"
    )
    step_num = 8 if args.estimator in ["dr-cpo", "mrdr", "tmle"] else 7
    print(f"\n{step_num}. Generating visualizations in {plot_dir}/...")
    plot_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Generate weight dashboards
        _generate_weight_dashboards(estimator, sampler, results, plot_dir, cal_result)

        # Generate calibration comparison
        _generate_calibration_comparison(
            dataset, calibrated_dataset, args, cal_result, plot_dir
        )

        # Generate policy estimates plot
        _generate_policy_estimates(dataset, results, args, summary_data, plot_dir)

        # Generate DR dashboard if applicable
        _generate_dr_dashboard(results, args, plot_dir)

        # Close all figures to free memory
        plt.close("all")

    except Exception as e:
        print(f"   ⚠️  Failed to generate some plots: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()


def _generate_weight_dashboards(
    estimator: Any,
    sampler: Any,
    results: Any,
    plot_dir: Path,
    cal_result: Optional[Any],
) -> None:
    """Generate weight dashboard visualizations.

    Args:
        estimator: Fitted estimator
        sampler: PrecomputedSampler
        results: EstimationResult object
        plot_dir: Directory to save plots
        cal_result: Optional calibration result
    """
    raw_weights_dict = {}
    calibrated_weights_dict = {}

    for policy in sampler.target_policies:
        weights = estimator.get_weights(policy)
        if weights is not None:
            calibrated_weights_dict[policy] = weights

        raw_weights = estimator.get_raw_weights(policy)
        if raw_weights is not None:
            raw_weights_dict[policy] = raw_weights
        elif weights is not None:
            raw_weights_dict[policy] = weights

    if raw_weights_dict and calibrated_weights_dict:
        # Generate combined overview dashboard (6-panel summary)
        fig, _ = plot_weight_dashboard_summary(
            raw_weights_dict,
            calibrated_weights_dict,
            n_samples=sampler.n_valid_samples,
            save_path=plot_dir / "weight_dashboard",
            diagnostics=results.diagnostics,
        )
        print(f"   ✓ Weight dashboard → {plot_dir}/weight_dashboard.png")
        plt.close(fig)

        # Generate per-policy detailed dashboard
        fig, _ = plot_weight_dashboard_detailed(
            raw_weights_dict,
            calibrated_weights_dict,
            n_samples=sampler.n_valid_samples,
            save_path=plot_dir / "weight_dashboard_per_policy",
            diagnostics=results.diagnostics,
            sampler=sampler,
            reward_calibrator=cal_result.calibrator if cal_result else None,
        )
        print(f"   ✓ Per-policy dashboard → {plot_dir}/weight_dashboard_per_policy.png")
        plt.close(fig)


def _generate_calibration_comparison(
    dataset: Any,
    calibrated_dataset: Any,
    args: Any,
    cal_result: Optional[Any],
    plot_dir: Path,
) -> None:
    """Generate calibration comparison plot.

    Args:
        dataset: Original dataset
        calibrated_dataset: Dataset with calibrated rewards
        args: Command-line arguments
        cal_result: Optional calibration result
        plot_dir: Directory to save plots
    """
    # Use the ORIGINAL dataset (before masking) to get all judge/oracle pairs
    # All samples in the logged dataset are from the base policy by definition
    # (fresh draws would be in separate FreshDrawDataset objects)
    judge_scores = []
    oracle_labels = []
    for s in dataset.samples:
        # All logged samples are base policy samples
        if args.judge_field in s.metadata and args.oracle_field in s.metadata:
            j_score = s.metadata.get(args.judge_field)
            o_label = s.metadata.get(args.oracle_field)
            if j_score is not None and o_label is not None:
                judge_scores.append(j_score)
                oracle_labels.append(o_label)

    if judge_scores and oracle_labels:
        # Get calibrated rewards if calibration was performed
        calibrated_preds = None
        if cal_result is not None:  # Calibration was performed
            # Get rewards for ALL samples (calibration was done on partial oracle labels)
            calibrated_preds_list = []
            for s_orig, s_cal in zip(dataset.samples, calibrated_dataset.samples):
                # Match samples from original dataset (with all oracle labels)
                # to calibrated dataset (with rewards)
                if (
                    args.judge_field in s_orig.metadata
                    and args.oracle_field in s_orig.metadata
                    and s_cal.reward is not None
                ):
                    calibrated_preds_list.append(s_cal.reward)

            # Use calibrated scores if we have them for all samples
            if len(calibrated_preds_list) == len(judge_scores):
                calibrated_preds = calibrated_preds_list

        fig = plot_calibration_comparison(
            judge_scores=np.array(judge_scores),
            oracle_labels=np.array(oracle_labels),
            calibrated_scores=(
                np.array(calibrated_preds) if calibrated_preds else None
            ),
            calibrator=cal_result.calibrator if cal_result else None,
            save_path=plot_dir / "calibration_comparison",
        )
        print(f"   ✓ Calibration comparison → {plot_dir}/calibration_comparison.png")
        plt.close(fig)


def _generate_policy_estimates(
    dataset: Any,
    results: Any,
    args: Any,
    summary_data: Dict[str, Any],
    plot_dir: Path,
) -> None:
    """Generate policy estimates forest plot.

    Args:
        dataset: Original dataset
        results: EstimationResult object
        args: Command-line arguments
        summary_data: Summary statistics from results
        plot_dir: Directory to save plots
    """
    policy_estimates = {"base": summary_data["base_mean"]}
    policy_ses = {"base": summary_data["base_se"]}

    for policy, estimate, se in zip(
        summary_data["target_policies"], results.estimates, results.standard_errors
    ):
        if not np.isnan(estimate):
            policy_estimates[policy] = estimate
            policy_ses[policy] = se

    # Try to get oracle values
    oracle_values = None
    if args.oracle_field in dataset.samples[0].metadata:
        oracle_values = load_oracle_ground_truth(
            args, dataset, summary_data["target_policies"]
        )

    fig = plot_policy_estimates(
        estimates=policy_estimates,
        standard_errors=policy_ses,
        oracle_values=oracle_values,
        base_policy="base",
        save_path=plot_dir / "policy_estimates",
    )
    print(f"   ✓ Policy estimates → {plot_dir}/policy_estimates.png")
    plt.close(fig)


def _generate_dr_dashboard(
    results: Any,
    args: Any,
    plot_dir: Path,
) -> None:
    """Generate DR dashboard if applicable.

    Args:
        results: EstimationResult object
        args: Command-line arguments
        plot_dir: Directory to save plots
    """
    if (
        args.estimator in ["dr-cpo", "mrdr", "tmle"]
        and "dr_diagnostics" in results.metadata
    ):
        try:
            fig, _ = plot_dr_dashboard(results)
            fig.savefig(plot_dir / "dr_dashboard.png", dpi=150, bbox_inches="tight")
            print(f"   ✓ DR dashboard → {plot_dir}/dr_dashboard.png")
            plt.close(fig)

        except Exception as e:
            print(f"   ⚠️  Could not generate DR dashboard: {e}")
