"""Estimator creation and configuration for CJE analysis.

This module handles creating the appropriate estimator (IPS, DR, TMLE, etc.)
based on configuration and adding fresh draws for DR-based estimators.

Following CLAUDE.md: Do one thing well - this module only handles estimation setup.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import sys

from cje.estimators import CalibratedIPS
from cje.estimators.dr_base import DRCPOEstimator
from cje.estimators.mrdr import MRDREstimator
from cje.estimators.tmle import TMLEEstimator
from cje.estimators.stacking import StackedDREstimator
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.calibration.dataset import calibrate_dataset
from cje.data.fresh_draws import load_fresh_draws_auto

# Note: validation import removed - function doesn't exist


def create_estimator(
    args: Any,
    sampler: PrecomputedSampler,
    calibrated_dataset: Any,
    cal_result: Optional[Any] = None,
) -> Union[
    CalibratedIPS, DRCPOEstimator, MRDREstimator, TMLEEstimator, StackedDREstimator
]:
    """Create the appropriate estimator based on args.

    Args:
        args: Command-line arguments
        sampler: PrecomputedSampler with data
        calibrated_dataset: Dataset with rewards
        cal_result: Optional calibration result

    Returns:
        Configured estimator instance
    """
    estimator_config = args.estimator_config or {}

    if args.estimator == "calibrated-ips":
        refuse_unreliable = estimator_config.get("refuse_unreliable", False)
        return CalibratedIPS(sampler, refuse_unreliable=refuse_unreliable)

    elif args.estimator == "raw-ips":
        clip_weight = estimator_config.get("clip_weight", 1e10)
        # Use CalibratedIPS with calibrate_weights=False for raw IPS
        return CalibratedIPS(sampler, calibrate_weights=False, clip_weight=clip_weight)

    elif args.estimator == "dr-cpo":
        return _create_dr_cpo(args, sampler, cal_result, estimator_config)

    elif args.estimator == "mrdr":
        return _create_mrdr(
            args, sampler, calibrated_dataset, cal_result, estimator_config
        )

    elif args.estimator == "tmle":
        return _create_tmle(
            args, sampler, calibrated_dataset, cal_result, estimator_config
        )

    elif args.estimator == "stacked-dr":
        return _create_stacked_dr(
            args, sampler, calibrated_dataset, cal_result, estimator_config
        )

    else:
        raise ValueError(f"Unknown estimator: {args.estimator}")


def _create_dr_cpo(
    args: Any,
    sampler: PrecomputedSampler,
    cal_result: Optional[Any],
    estimator_config: Dict[str, Any],
) -> DRCPOEstimator:
    """Create DR-CPO estimator."""
    n_folds = estimator_config.get("n_folds", 5)

    if cal_result and cal_result.calibrator:
        dr_estimator = DRCPOEstimator(
            sampler,
            n_folds=n_folds,
            reward_calibrator=cal_result.calibrator,
        )
        print("   Using CalibratorBackedOutcomeModel (reusing calibration models)")
    else:
        dr_estimator = DRCPOEstimator(sampler, n_folds=n_folds)
        print("   Using IsotonicOutcomeModel (refitting models)")

    # Load fresh draws
    print("   Loading fresh draws for DR estimation...")
    add_fresh_draws(dr_estimator, args, sampler, estimator_config)

    return dr_estimator


def _create_mrdr(
    args: Any,
    sampler: PrecomputedSampler,
    calibrated_dataset: Any,
    cal_result: Optional[Any],
    estimator_config: Dict[str, Any],
) -> MRDREstimator:
    """Create MRDR estimator."""
    n_folds = estimator_config.get("n_folds", 5)
    omega_mode = estimator_config.get("omega_mode", "snips")

    # MRDR works best with cross-fitted calibration
    if args.oracle_coverage < 1.0 and (not cal_result or not cal_result.calibrator):
        print("   ⚠️  MRDR works best with cross-fitted calibration. Re-calibrating...")
        # Note: validation check removed - function doesn't exist
        calibrated_dataset, cal_result = calibrate_dataset(
            calibrated_dataset,
            judge_field="judge_score",
            oracle_field="oracle_label",
            enable_cross_fit=True,
            n_folds=n_folds,
        )
        sampler = PrecomputedSampler(calibrated_dataset)

    mrdr_estimator = MRDREstimator(sampler, n_folds=n_folds, omega_mode=omega_mode)
    print(f"   Using MRDR with omega_mode='{omega_mode}'")

    # Load fresh draws
    print("   Loading fresh draws for MRDR estimation...")
    add_fresh_draws(mrdr_estimator, args, sampler, estimator_config)

    return mrdr_estimator


def _create_tmle(
    args: Any,
    sampler: PrecomputedSampler,
    calibrated_dataset: Any,
    cal_result: Optional[Any],
    estimator_config: Dict[str, Any],
) -> TMLEEstimator:
    """Create TMLE estimator."""
    n_folds = estimator_config.get("n_folds", 5)
    link = estimator_config.get("link", "logit")

    # TMLE works best with cross-fitted calibration
    if args.oracle_coverage < 1.0 and (not cal_result or not cal_result.calibrator):
        print("   ⚠️  TMLE works best with cross-fitted calibration. Re-calibrating...")
        # Note: validation check removed - function doesn't exist
        calibrated_dataset, cal_result = calibrate_dataset(
            calibrated_dataset,
            judge_field="judge_score",
            oracle_field="oracle_label",
            enable_cross_fit=True,
            n_folds=n_folds,
        )
        sampler = PrecomputedSampler(calibrated_dataset)

    tmle_estimator = TMLEEstimator(sampler, n_folds=n_folds, link=link)
    print(f"   Using TMLE with link='{link}'")

    # Load fresh draws
    print("   Loading fresh draws for TMLE estimation...")
    add_fresh_draws(tmle_estimator, args, sampler, estimator_config)

    return tmle_estimator


def add_fresh_draws(
    estimator: Any,
    args: Any,
    sampler: PrecomputedSampler,
    estimator_config: Dict[str, Any],
) -> None:
    """Add fresh draws to a DR estimator for all target policies.

    NOTE: As of the latest version, DR estimators auto-load fresh draws
    when estimate() is called. This function is kept for backward compatibility
    but is no longer necessary.

    Args:
        estimator: DR estimator instance
        args: Command-line arguments (contains data path)
        sampler: PrecomputedSampler with target policies
        estimator_config: Estimator configuration (unused but kept for compatibility)

    Raises:
        FileNotFoundError: If fresh draws are not available
    """
    # Auto-loading is now handled internally by DR estimators
    # This function is kept for backward compatibility but does nothing
    return


def _create_stacked_dr(
    args: Any,
    sampler: PrecomputedSampler,
    calibrated_dataset: Any,
    cal_result: Optional[Any],
    estimator_config: Dict[str, Any],
) -> StackedDREstimator:
    """Create stacked DR estimator combining DR-CPO, TMLE, and MRDR."""
    n_folds = estimator_config.get("n_folds", 5)
    estimators = estimator_config.get("estimators", ["dr-cpo", "tmle", "mrdr"])

    # Create stacked estimator
    if cal_result and cal_result.calibrator:
        stacked_estimator = StackedDREstimator(
            sampler,
            reward_calibrator=cal_result.calibrator,
            estimators=estimators,
            n_folds=n_folds,
        )
        print(f"   Creating stacked estimator with: {', '.join(estimators)}")
        print("   Using CalibratorBackedOutcomeModel (reusing calibration models)")
    else:
        stacked_estimator = StackedDREstimator(
            sampler,
            estimators=estimators,
            n_folds=n_folds,
        )
        print(f"   Creating stacked estimator with: {', '.join(estimators)}")
        print("   Using IsotonicOutcomeModel (refitting models)")

    # Load fresh draws
    print("   Loading fresh draws for stacked DR estimation...")
    add_fresh_draws(stacked_estimator, args, sampler, estimator_config)

    return stacked_estimator
