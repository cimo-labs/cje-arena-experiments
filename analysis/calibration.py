"""Calibration and reward assignment for CJE analysis.

This module handles calibrating judge scores to oracle labels and assigning
rewards to samples. It supports partial oracle coverage and cross-fitting.

Following CLAUDE.md: Do one thing well - this module only handles calibration.
"""

import random
import numpy as np
from typing import Any, Optional, Tuple, Set
from cje.calibration.dataset import calibrate_dataset


def handle_rewards(
    dataset: Any, args: Any, analysis_config: dict, verbose: bool = True
) -> Tuple[Any, Optional[Any]]:
    """Handle reward assignment and calibration.

    This function determines whether to:
    1. Use existing rewards from the dataset
    2. Use oracle labels directly as rewards
    3. Calibrate judge scores to get rewards

    Args:
        dataset: Input dataset
        args: Command-line arguments
        analysis_config: Configuration dictionary
        verbose: Whether to print status

    Returns:
        Tuple of (calibrated_dataset, calibration_result)
    """
    if verbose:
        print("\n2. Handling rewards...")

    calibrated_dataset = None
    cal_result = None

    # Check if rewards already exist
    rewards_exist = sum(1 for s in dataset.samples if s.reward is not None)

    if rewards_exist > 0:
        # Use existing rewards
        if verbose:
            print(f"   ✓ Using {rewards_exist} pre-computed rewards from dataset")
        calibrated_dataset = dataset

        # Check if we need cross-fitted models for SIMCal
        if _needs_crossfit_models(dataset, args):
            cal_result = _fit_crossfit_models(dataset, args, analysis_config, verbose)
            _add_fold_ids(calibrated_dataset, cal_result)

    elif args.use_oracle or args.oracle_coverage == 1.0:
        # Use oracle labels directly
        calibrated_dataset = _use_oracle_as_rewards(dataset, args, verbose)

        # Still fit cross-fitted models for SIMCal ordering
        if verbose:
            print("   Fitting cross-fitted models for SIMCal ordering index...")
        _, cal_result = calibrate_dataset(
            dataset,
            judge_field=args.judge_field,
            oracle_field=args.oracle_field,
            enable_cross_fit=True,
            n_folds=analysis_config["n_folds"],
        )
        _add_fold_ids(calibrated_dataset, cal_result)

        if verbose and cal_result:
            print(
                f"   ✓ Cross-fitted models ready (RMSE: {cal_result.calibration_rmse:.3f})"
            )

    else:
        # Calibrate with partial oracle coverage
        calibrated_dataset, cal_result = _calibrate_with_coverage(
            dataset, args, analysis_config, verbose
        )

    return calibrated_dataset, cal_result


def _needs_crossfit_models(dataset: Any, args: Any) -> bool:
    """Check if we need to fit cross-fitted models."""
    has_fold_ids = all("cv_fold" in s.metadata for s in dataset.samples[:10])
    has_oracle = args.oracle_field in dataset.samples[0].metadata
    return not has_fold_ids and has_oracle


def _fit_crossfit_models(
    dataset: Any, args: Any, analysis_config: dict, verbose: bool
) -> Optional[Any]:
    """Fit cross-fitted calibration models."""
    if verbose:
        print("   Fitting cross-fitted models for SIMCal ordering index...")

    try:
        _, cal_result = calibrate_dataset(
            dataset,
            judge_field=args.judge_field,
            oracle_field=args.oracle_field,
            enable_cross_fit=True,
            n_folds=analysis_config["n_folds"],
        )
        if verbose:
            print(f"   ✓ Cross-fitted models ready")
        return cal_result
    except Exception as e:
        if verbose:
            print(f"   ⚠️  Could not fit cross-fitted models: {e}")
        return None


def _add_fold_ids(dataset: Any, cal_result: Any) -> None:
    """Legacy function - no longer adds fold IDs to metadata.

    Folds are now computed on-demand from prompt_id using the unified fold system.
    This function is kept for backward compatibility but does nothing.
    """
    # Folds are computed dynamically via cje.data.folds.get_fold()
    # No need to store in metadata anymore
    pass


def _use_oracle_as_rewards(dataset: Any, args: Any, verbose: bool) -> Any:
    """Use oracle labels directly as rewards."""
    if verbose:
        print("   Using oracle labels directly as rewards...")

    oracle_count = 0
    for sample in dataset.samples:
        if args.oracle_field in sample.metadata:
            sample.reward = float(sample.metadata[args.oracle_field])
            oracle_count += 1

    if verbose:
        print(f"   ✓ Assigned {oracle_count} oracle labels as rewards")

    return dataset


def _calibrate_with_coverage(
    dataset: Any, args: Any, analysis_config: dict, verbose: bool
) -> Tuple[Any, Any]:
    """Calibrate with partial oracle coverage."""
    if verbose:
        print(f"   Calibrating with {args.oracle_coverage:.0%} oracle coverage...")

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Store original oracle labels (will be restored later for visualization)
    original_oracle_labels = {}

    # Mask some oracle labels if partial coverage
    if args.oracle_coverage < 1.0:
        original_oracle_labels = _mask_oracle_labels(
            dataset, args, args.oracle_coverage
        )

    # Calibrate with cross-fitting for DR
    calibrated_dataset, cal_result = calibrate_dataset(
        dataset,
        judge_field=args.judge_field,
        oracle_field=args.oracle_field,
        enable_cross_fit=True,
        n_folds=analysis_config["n_folds"],
    )

    if verbose:
        print(f"   ✓ Calibrated using {cal_result.n_oracle} oracle labels")
        print(f"   ✓ Calibration RMSE: {cal_result.calibration_rmse:.3f}")

    # Store original labels in metadata for later restoration
    calibrated_dataset.metadata["_original_oracle_labels"] = original_oracle_labels
    # Also store on original dataset so it can be restored there too
    dataset.metadata["_original_oracle_labels"] = original_oracle_labels

    return calibrated_dataset, cal_result


def _mask_oracle_labels(dataset: Any, args: Any, oracle_coverage: float) -> dict:
    """Mask oracle labels to simulate partial coverage.

    Returns:
        Dictionary mapping sample index to original oracle value
    """
    samples_with_oracle = [
        i
        for i, s in enumerate(dataset.samples)
        if args.oracle_field in s.metadata and s.metadata[args.oracle_field] is not None
    ]

    n_keep = max(2, int(len(samples_with_oracle) * oracle_coverage))
    keep_indices = set(
        random.sample(samples_with_oracle, min(n_keep, len(samples_with_oracle)))
    )

    original_oracle_labels = {}
    for i, sample in enumerate(dataset.samples):
        if i not in keep_indices and args.oracle_field in sample.metadata:
            original_oracle_labels[i] = sample.metadata[args.oracle_field]
            sample.metadata[args.oracle_field] = None

    return original_oracle_labels


def restore_oracle_labels(dataset: Any, args: Any) -> None:
    """Restore masked oracle labels for visualization.

    This should be called after estimation but before visualization.
    """
    if hasattr(dataset, "_original_oracle_labels"):
        for i, oracle_value in dataset._original_oracle_labels.items():
            dataset.samples[i].metadata[args.oracle_field] = oracle_value
