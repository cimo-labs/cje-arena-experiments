"""Base class for ablation experiments."""

import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from scipy import stats

from .schemas import ExperimentSpec, create_result

# Handle import based on execution context
try:
    from ..config import DR_CONFIG
except ImportError:
    # Fallback for different import paths
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import DR_CONFIG as DR_CONFIG  # type: ignore[import, no-redef]

# No local diagnostics file needed!
# Import standard diagnostics from CJE
from cje.diagnostics.weights import effective_sample_size, hill_tail_index


# Local function for weight CV (not in CJE)
def weight_cv(weights: np.ndarray) -> float:
    """Coefficient of variation of weights."""
    weights = np.asarray(weights)
    weights = weights[np.isfinite(weights)]
    if len(weights) == 0:
        return float(np.nan)
    mean_w = np.mean(weights)
    if mean_w == 0:
        return float(np.nan)
    return float(np.std(weights) / mean_w)


from cje import load_dataset_from_jsonl
from cje.calibration import calibrate_dataset
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.estimators import CalibratedIPS
from cje.estimators.stacking import StackedDREstimator
from cje.estimators.dr_base import DRCPOEstimator
from cje.estimators.orthogonalized_calibrated_dr import OrthogonalizedCalibratedDRCPO
from cje.estimators.orthogonalized_ips import OrthogonalizedCalibratedIPS
from cje.estimators.tr_cpo import TRCPOEstimator
from cje.estimators.mrdr import MRDREstimator
from cje.estimators.tmle import TMLEEstimator
from cje.estimators.direct_method import CalibratedDirectEstimator
from cje.data.fresh_draws import load_fresh_draws_auto

logger = logging.getLogger(__name__)


class BaseAblation:
    """Base class for all ablation experiments.

    Provides common functionality:
    - Data loading and preparation
    - Oracle masking and calibration
    - Estimator creation and execution
    - Diagnostic computation
    """

    def __init__(self, name: str):
        """Initialize ablation.

        Args:
            name: Name of this ablation (e.g., "oracle_coverage")
        """
        self.name = name
        self.results: List[Dict[str, Any]] = []

    def prepare_dataset(
        self, spec: ExperimentSpec, seed: int
    ) -> Tuple[Any, int, Dict[int, Any]]:
        """Load and prepare dataset with oracle masking.

        Args:
            spec: Experiment specification
            seed: Random seed

        Returns:
            (dataset, n_oracle, original_oracle_labels)
        """
        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)

        # Load dataset
        dataset = load_dataset_from_jsonl(spec.dataset_path)

        # Store the dataset path in metadata for auto-loading fresh draws
        dataset.metadata["dataset_path"] = spec.dataset_path

        # Subsample if requested
        if spec.sample_size is not None:
            n_samples = min(spec.sample_size, len(dataset.samples))
            indices = sorted(random.sample(range(len(dataset.samples)), n_samples))
            dataset.samples = [dataset.samples[i] for i in indices]
        elif spec.sample_fraction is not None:
            n_samples = int(len(dataset.samples) * spec.sample_fraction)
            indices = sorted(random.sample(range(len(dataset.samples)), n_samples))
            dataset.samples = [dataset.samples[i] for i in indices]

        # Mask oracle labels if coverage < 1
        original_oracle_labels = {}
        n_oracle = len(dataset.samples)  # Default: all have oracle

        if spec.oracle_coverage is not None and spec.oracle_coverage < 1.0:
            # Find samples with oracle labels (top-level field, not in metadata)
            oracle_indices = [
                i
                for i, s in enumerate(dataset.samples)
                if s.oracle_label is not None
            ]

            # Determine how many to keep
            n_keep = max(2, int(len(oracle_indices) * spec.oracle_coverage))
            keep_indices = set(
                random.sample(oracle_indices, min(n_keep, len(oracle_indices)))
            )

            # Mask labels not in keep set (oracle_label is a top-level field)
            for i, sample in enumerate(dataset.samples):
                if i not in keep_indices and sample.oracle_label is not None:
                    original_oracle_labels[i] = sample.oracle_label
                    sample.oracle_label = None

            n_oracle = len(keep_indices)

        return dataset, n_oracle, original_oracle_labels

    def create_estimator(
        self, spec: ExperimentSpec, sampler: PrecomputedSampler, cal_result: Any, seed: int = 42
    ) -> Any:
        """Create estimator based on specification.

        Args:
            spec: Experiment specification
            sampler: PrecomputedSampler with data
            cal_result: Calibration result
            seed: Random seed for reproducibility

        Returns:
            Estimator instance
        """
        # Extract settings from spec.extra
        # Note: use_iic was removed from library - no longer supported
        use_weight_calibration = (
            spec.extra.get("use_weight_calibration", False) if spec.extra else False
        )
        # Propagate oracle-calibrator uncertainty into SEs (ON by default unless explicitly disabled)
        oua = spec.extra.get("oua_jackknife", True) if spec.extra else True
        # Variance budget (rho) for SIMCal - default 1.0 means no variance increase
        var_cap = spec.extra.get("var_cap", 1.0) if spec.extra else 1.0

        # Log parameter settings if needed
        # logger.info(f"Creating {spec.estimator} with use_iic={use_iic}, "
        #            f"use_weight_calibration(SIMCal)={use_weight_calibration}")

        # Stacking MC config knobs from spec.extra (optional)
        include_mc = (
            bool(spec.extra.get("include_mc_in_objective", True))
            if spec.extra
            else True
        )
        mc_lambda = float(spec.extra.get("mc_lambda", 1.0)) if spec.extra else 1.0

        estimator_map = {
            "direct": lambda s: CalibratedDirectEstimator(
                target_policies=list(s.target_policies),
                reward_calibrator=cal_result.calibrator if cal_result else None,
                run_diagnostics=True,
                oua_jackknife=oua,
            ),
            "naive-direct": lambda s: CalibratedDirectEstimator(
                target_policies=list(s.target_policies),
                reward_calibrator=None,  # Force no calibration
                run_diagnostics=True,
                oua_jackknife=False,  # No oracle uncertainty (no calibration)
            ),
            "raw-ips": lambda s: CalibratedIPS(
                s,
                calibrate_weights=False,
                reward_calibrator=cal_result.calibrator if cal_result else None,
                oua_jackknife=oua,
                use_outer_cv=True,
                n_outer_folds=5,
                outer_cv_seed=seed,
            ),  # No weight calibration, but still support OUA
            "calibrated-ips": lambda s: CalibratedIPS(
                s,
                calibrate_weights=True,
                reward_calibrator=cal_result.calibrator if cal_result else None,
                oua_jackknife=oua,  # Always use calibration, enable OUA
                use_outer_cv=True,  # Enable outer CV for robust inference
                n_outer_folds=5,  # Use 5 folds for clustering
                outer_cv_seed=seed,  # Use experiment seed for reproducibility
                var_cap=var_cap,  # Pass variance budget (rho) parameter
            ),
            "orthogonalized-ips": lambda s: OrthogonalizedCalibratedIPS(
                s,
                reward_calibrator=cal_result.calibrator if cal_result else None,
                oua_jackknife=oua,
                use_outer_cv=True,  # Enable outer CV for robust inference
                n_outer_folds=5,  # Use 5 folds for clustering
                outer_cv_seed=seed,  # Use experiment seed for reproducibility
                var_cap=var_cap,  # Pass variance budget (rho) parameter
            ),
            "dr-cpo": lambda s: DRCPOEstimator(
                s,
                reward_calibrator=cal_result.calibrator if cal_result else None,
                n_folds=DR_CONFIG["n_folds"],
                use_calibrated_weights=use_weight_calibration,  # Controlled by use_weight_calibration flag
                oua_jackknife=oua,
                var_cap=var_cap,  # Pass variance budget (rho) parameter
            ),
            "oc-dr-cpo": lambda s: OrthogonalizedCalibratedDRCPO(
                s,
                reward_calibrator=cal_result.calibrator if cal_result else None,
                n_folds=DR_CONFIG["n_folds"],
                use_calibrated_weights=use_weight_calibration,  # Controlled by use_weight_calibration flag
                oua_jackknife=oua,
                # Note: var_cap is handled by sampler, not by OC-DR-CPO directly
            ),
            "tr-cpo-e": lambda s: TRCPOEstimator(
                s,
                reward_calibrator=cal_result.calibrator if cal_result else None,
                n_folds=DR_CONFIG["n_folds"],
                oua_jackknife=oua,
                use_efficient_tr=True,  # Efficient TR-CPO uses m̂(S)
            ),
            "tr-cpo-e-anchored-orthogonal": lambda s: TRCPOEstimator(
                s,
                reward_calibrator=cal_result.calibrator if cal_result else None,
                n_folds=DR_CONFIG["n_folds"],
                oua_jackknife=oua,
                use_efficient_tr=True,  # Efficient TR-CPO uses m̂(S)
                anchor_on_simcal=True,  # Anchor on SIMCal weights for stability
                add_orthogonalizer=True,  # Add orthogonalizer for variance reduction
            ),
            "calibrated-dr-cpo": lambda s: DRCPOEstimator(
                s,
                reward_calibrator=cal_result.calibrator if cal_result else None,
                n_folds=DR_CONFIG["n_folds"],
                use_calibrated_weights=True,  # Use SIMCal calibrated weights
                oua_jackknife=oua,
                var_cap=var_cap,  # Pass variance budget (rho) parameter
            ),
            "mrdr": lambda s: MRDREstimator(
                s,
                reward_calibrator=cal_result.calibrator if cal_result else None,
                n_folds=DR_CONFIG["n_folds"],
                use_calibrated_weights=use_weight_calibration,  # Controlled by use_weight_calibration flag
                oua_jackknife=oua,
                var_cap=var_cap,  # Pass variance budget (rho) parameter
            ),
            "tmle": lambda s: TMLEEstimator(
                s,
                reward_calibrator=cal_result.calibrator if cal_result else None,
                n_folds=DR_CONFIG["n_folds"],
                use_calibrated_weights=use_weight_calibration,  # Controlled by use_weight_calibration flag
                oua_jackknife=oua,
                var_cap=var_cap,  # Pass variance budget (rho) parameter
            ),
            "stacked-dr": lambda s: StackedDREstimator(
                s,
                # Uses default estimators: ["dr-cpo", "tmle", "mrdr"]
                reward_calibrator=cal_result.calibrator if cal_result else None,
                n_folds=DR_CONFIG["n_folds"],  # Use n_folds, not V_folds
                use_calibrated_weights=use_weight_calibration,  # Controlled by use_weight_calibration flag
                oua_jackknife=oua,
                covariance_regularization=1e-4,  # Add regularization for numerical stability
                include_mc_in_objective=include_mc,
                mc_lambda=mc_lambda,
                var_cap=var_cap,  # Pass variance budget (rho) parameter
                # Remove use_outer_split - it doesn't exist
            ),
        }

        if spec.estimator not in estimator_map:
            raise ValueError(f"Unknown estimator: {spec.estimator}")

        return estimator_map[spec.estimator](sampler)

    def _compute_rmse(
        self, estimates: Dict[str, float], truths: Dict[str, float]
    ) -> float:
        """Compute RMSE between estimates and oracle truths.

        Note: The 'unhelpful' policy is excluded from RMSE calculation because
        it has a very different reward distribution (mean ~0.14) compared to
        other policies (mean ~0.76). This causes systematic calibration bias.
        """
        if not estimates or not truths:
            return float(np.nan)

        squared_errors = []
        for policy in estimates:
            # Skip unhelpful policy - different reward distribution
            if policy == "unhelpful":
                continue

            if policy in truths:
                est = estimates[policy]
                truth = truths[policy]
                if np.isfinite(est) and np.isfinite(truth):
                    squared_errors.append((est - truth) ** 2)

        if not squared_errors:
            return float(np.nan)

        return float(np.sqrt(np.mean(squared_errors)))

    def _load_oracle_ground_truth(
        self, dataset_path: str, dataset: Any, target_policies: List[str]
    ) -> Dict[str, float]:
        """Load oracle ground truth values for comparison.

        Args:
            dataset_path: Path to dataset file
            dataset: Dataset object
            target_policies: List of target policies

        Returns:
            Dictionary mapping policy names to oracle mean values
        """
        oracle_means = {}
        data_dir = Path(dataset_path).parent
        responses_dir = data_dir / "responses"

        # Load base policy oracle labels from dataset (top-level field, not metadata)
        base_oracle_values = []
        for sample in dataset.samples:
            if hasattr(sample, "oracle_label") and sample.oracle_label is not None:
                base_oracle_values.append(sample.oracle_label)

        if base_oracle_values:
            oracle_means["base"] = float(np.mean(base_oracle_values))

        # Load oracle labels for each target policy from response files
        for policy in target_policies:
            response_file = responses_dir / f"{policy}_responses.jsonl"
            if response_file.exists():
                oracle_values = []
                with open(response_file, "r") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if (
                                "metadata" in data
                                and "oracle_label" in data["metadata"]
                            ):
                                oracle_val = data["metadata"]["oracle_label"]
                                if oracle_val is not None:
                                    oracle_values.append(oracle_val)
                        except json.JSONDecodeError:
                            continue

                if oracle_values:
                    oracle_means[policy] = float(np.mean(oracle_values))
                    # Store per-policy oracle counts for debiasing
                    try:
                        dataset_name = getattr(self, "name", "ablation")
                    except Exception:
                        dataset_name = "ablation"
                    # Attach to an attribute for later consumption in run_single
                    if not hasattr(self, "_oracle_counts_per_policy"):
                        self._oracle_counts_per_policy = {}
                    self._oracle_counts_per_policy[policy] = int(len(oracle_values))

        return oracle_means

    # CF-bits metrics removed from library - method deleted

    def compute_diagnostics(
        self, estimator: Any, result: Dict[str, Any], n_total: int
    ) -> None:
        """Compute and add diagnostics to result.

        Args:
            estimator: Fitted estimator
            result: Result dictionary to update
            n_total: Total number of samples
        """
        # Import CJE's diagnostic functions
        from cje.diagnostics.overlap import compute_overlap_metrics
        from cje.diagnostics.weights import hill_tail_index_stable

        # Get target policies
        policies = estimator.sampler.target_policies

        for policy in policies:
            try:
                # Get weights (method may vary by estimator)
                if hasattr(estimator, "get_weights"):
                    weights = estimator.get_weights(policy)
                elif hasattr(estimator, "_weights_cache"):
                    weights = estimator._weights_cache.get(policy)
                else:
                    weights = estimator.sampler.compute_importance_weights(policy)

                if weights is not None and len(weights) > 0:
                    # Compute basic diagnostics using local functions (simpler, direct)
                    ess = effective_sample_size(weights)
                    result["ess_absolute"][policy] = ess
                    result["ess_relative"][policy] = (
                        100.0 * ess / n_total if n_total > 0 else 0
                    )
                    result["tail_alpha"][policy] = hill_tail_index(weights)
                    result["weight_cv"][policy] = weight_cv(weights)

                    # Max weight (normalized)
                    weights_norm = weights / np.sum(weights)
                    result["max_weight"][policy] = float(np.max(weights_norm))

                    # Mass concentration (fraction of weights near zero)
                    # Use threshold of 1/(10*n) as "near zero"
                    threshold = 1.0 / (10 * len(weights))
                    result["mass_concentration"][policy] = float(
                        np.mean(weights_norm < threshold)
                    )

                    # Compute advanced overlap metrics from CJE diagnostics
                    overlap_metrics = compute_overlap_metrics(
                        weights,
                        target_ci_halfwidth=0.01,
                        n_samples=n_total,
                        compute_tail_index=True,
                        auto_tune_threshold=False,
                    )

                    # Store additional diagnostics
                    result.setdefault("hellinger_affinity", {})[policy] = float(
                        overlap_metrics.hellinger_affinity
                    )
                    result.setdefault("overlap_quality", {})[
                        policy
                    ] = overlap_metrics.overlap_quality
                    result.setdefault("can_calibrate", {})[policy] = bool(
                        overlap_metrics.can_calibrate
                    )
                    result.setdefault("recommended_method", {})[
                        policy
                    ] = overlap_metrics.recommended_method

            except Exception as e:
                logger.warning(f"Failed to compute diagnostics for {policy}: {e}")

            # CF-bits computation removed from library

    def run_single(self, spec: ExperimentSpec, seed: int) -> Dict[str, Any]:
        """Run single experiment with given seed.

        Args:
            spec: Experiment specification
            seed: Random seed

        Returns:
            Result dictionary
        """
        # Create result
        result = create_result(spec, seed)

        try:
            # Prepare data
            dataset, n_oracle, original_oracle_labels = self.prepare_dataset(spec, seed)
            result["n_samples"] = len(dataset.samples)
            result["n_oracle"] = n_oracle
            result["oracle_slice_size"] = (
                n_oracle  # Also store as oracle_slice_size for clarity
            )

            # Check if we should use covariates (needed by both calibration and fresh draws)
            use_covariates = (
                spec.extra.get("use_covariates", False) if spec.extra else False
            )

            # Skip calibration for naive-direct (uses raw judge scores)
            # All other estimators use reward calibration (judge → oracle)
            if spec.estimator == "naive-direct":
                # Manually populate reward field with raw judge scores
                # (calibrate_dataset would normally do this with calibrated values)
                for sample in dataset.samples:
                    # judge_score is a direct field on Sample, not in metadata
                    if sample.judge_score is not None:
                        sample.reward = float(sample.judge_score)
                    else:
                        # If no judge score, set reward to None (will be filtered)
                        sample.reward = None
                calibrated_dataset = dataset
                cal_result = None
            else:
                # Calibration mode can be configured via spec.extra["reward_calibration_mode"]
                reward_calibration_mode = (
                    spec.extra.get("reward_calibration_mode", "monotone")
                    if spec.extra
                    else "monotone"
                )

                calibrated_dataset, cal_result = calibrate_dataset(
                    dataset,
                    judge_field="judge_score",
                    oracle_field="oracle_label",
                    enable_cross_fit=True,
                    n_folds=DR_CONFIG["n_folds"] if n_oracle >= 50 else 3,
                    calibration_mode=reward_calibration_mode,
                    random_seed=seed,  # Pass the experiment seed for fold assignment
                    covariate_names=["response_length"] if use_covariates else None,
                )

            if cal_result:
                result["calibration_rmse"] = cal_result.calibration_rmse
                # R² may not be available in older versions
                if hasattr(cal_result, "calibration_r2"):
                    result["calibration_r2"] = cal_result.calibration_r2
                elif hasattr(cal_result, "r2"):
                    result["calibration_r2"] = cal_result.r2

                # Extract calibrated reward range to detect overlap issues
                # This catches the unhelpful policy problem (min ~0.4 instead of 0)
                if calibrated_dataset and calibrated_dataset.metadata:
                    cal_info = calibrated_dataset.metadata.get("calibration_info", {})
                    if "f_min" in cal_info:
                        result["calibrated_reward_min"] = cal_info["f_min"]
                    if "f_max" in cal_info:
                        result["calibrated_reward_max"] = cal_info["f_max"]

                    # Flag overlap issues: calibration can't extrapolate beyond observed range
                    # If min > 0.1 or max < 0.9, we have incomplete coverage of [0,1]
                    if "f_min" in cal_info and "f_max" in cal_info:
                        f_min = cal_info["f_min"]
                        f_max = cal_info["f_max"]
                        if f_min > 0.1 or f_max < 0.9:
                            result["reward_overlap_warning"] = True
                            logger.warning(
                                f"Calibrated reward range [{f_min:.3f}, {f_max:.3f}] "
                                f"does not cover full [0,1] oracle range - estimates may be biased"
                            )
                # Track which calibration mode was actually used
                if cal_result.calibrator and hasattr(
                    cal_result.calibrator, "selected_mode"
                ):
                    result["reward_calibration_used"] = (
                        cal_result.calibrator.selected_mode
                    )
                else:
                    result["reward_calibration_used"] = reward_calibration_mode

            # Create sampler and estimator
            sampler = PrecomputedSampler(calibrated_dataset)
            estimator = self.create_estimator(spec, sampler, cal_result, seed)

            # Add fresh draws for DR methods and Direct method
            if spec.estimator in [
                "direct",
                "naive-direct",
                "dr-cpo",
                "oc-dr-cpo",
                "calibrated-dr-cpo",
                "mrdr",
                "tmle",
                "tr-cpo-e",
                "tr-cpo-e-anchored-orthogonal",
                "stacked-dr",
            ]:
                data_dir = Path(spec.dataset_path).parent

                # Get prompt IDs from the subsampled dataset
                dataset_prompt_ids = set()
                for sample in calibrated_dataset.samples:
                    if hasattr(sample, "prompt_id") and sample.prompt_id:
                        dataset_prompt_ids.add(sample.prompt_id)

                # Get covariate names from calibration result if using covariates
                covariate_names = None
                if use_covariates and cal_result and hasattr(cal_result.calibrator, 'covariate_names'):
                    covariate_names = cal_result.calibrator.covariate_names or []

                for policy in sampler.target_policies:
                    try:
                        # Load ALL fresh draws
                        all_fresh_draws = load_fresh_draws_auto(
                            data_dir, policy, verbose=False
                        )

                        # Filter to only include fresh draws matching our subsampled prompts
                        if dataset_prompt_ids:
                            from cje.data.fresh_draws import (
                                FreshDrawSample,
                                FreshDrawDataset,
                                compute_response_covariates,
                            )

                            filtered_samples: List[FreshDrawSample] = []
                            for fd_sample in all_fresh_draws.samples:
                                if (
                                    hasattr(fd_sample, "prompt_id")
                                    and fd_sample.prompt_id in dataset_prompt_ids
                                ):
                                    filtered_samples.append(fd_sample)

                            # Create filtered fresh draws dataset with required fields

                            # Count draws per prompt
                            draws_per_prompt_dict: Dict[str, int] = {}
                            for fd_sample in filtered_samples:
                                prompt_id = (
                                    fd_sample.prompt_id
                                    if hasattr(fd_sample, "prompt_id")
                                    else None
                                )
                                if prompt_id:
                                    draws_per_prompt_dict[prompt_id] = (
                                        draws_per_prompt_dict.get(prompt_id, 0) + 1
                                    )

                            # Get the most common draws per prompt value
                            draws_per_prompt = (
                                max(
                                    set(draws_per_prompt_dict.values()),
                                    key=list(draws_per_prompt_dict.values()).count,
                                )
                                if draws_per_prompt_dict
                                else 10
                            )

                            filtered_fresh_draws = FreshDrawDataset(
                                samples=filtered_samples,
                                target_policy=policy,  # Use the policy we're processing
                                draws_per_prompt=draws_per_prompt,
                            )

                            # Compute covariates for fresh draws if needed
                            if covariate_names:
                                filtered_fresh_draws = compute_response_covariates(
                                    filtered_fresh_draws, covariate_names=covariate_names
                                )
                                logger.info(
                                    f"Computed covariates {covariate_names} for fresh draws (policy={policy})"
                                )

                            estimator.add_fresh_draws(policy, filtered_fresh_draws)
                            logger.info(
                                f"Added {len(filtered_samples)}/{len(all_fresh_draws.samples)} fresh draws for {policy}"
                            )
                        else:
                            # If no prompt IDs, use all fresh draws (fallback)
                            # Still compute covariates if needed
                            if covariate_names:
                                from cje.data.fresh_draws import compute_response_covariates
                                all_fresh_draws = compute_response_covariates(
                                    all_fresh_draws, covariate_names=covariate_names
                                )
                            estimator.add_fresh_draws(policy, all_fresh_draws)

                    except FileNotFoundError:
                        logger.warning(f"No fresh draws for {policy}")

            # Run estimation
            estimation_result = estimator.fit_and_estimate()

            # Extract results
            # Use EstimationResult.confidence_interval() method which properly handles:
            # - Degrees of freedom from metadata["degrees_of_freedom"]
            # - t-distribution when DF is available
            # - z-distribution fallback for large samples
            ci_lower, ci_upper = estimation_result.confidence_interval(alpha=0.05)

            for i, policy in enumerate(sampler.target_policies):
                result["estimates"][policy] = float(estimation_result.estimates[i])

                # Store standard errors if available
                if estimation_result.standard_errors is not None:
                    result["standard_errors"][policy] = float(estimation_result.standard_errors[i])

                # Store robust SEs separately if available (includes OUA)
                if (
                    hasattr(estimation_result, "robust_standard_errors")
                    and estimation_result.robust_standard_errors is not None
                ):
                    result.setdefault("robust_standard_errors", {})[policy] = float(
                        estimation_result.robust_standard_errors[i]
                    )

                # Store confidence intervals using the official method
                # This automatically uses t-distribution when degrees_of_freedom metadata is present
                result["confidence_intervals"][policy] = (
                    float(ci_lower[i]),
                    float(ci_upper[i]),
                )

                # Also store as robust CI if robust CIs are pre-computed
                if (
                    hasattr(estimation_result, "robust_confidence_intervals")
                    and estimation_result.robust_confidence_intervals is not None
                    and i < len(estimation_result.robust_confidence_intervals)
                ):
                    robust_ci = estimation_result.robust_confidence_intervals[i]
                    if isinstance(robust_ci, (list, tuple)) and len(robust_ci) == 2:
                        result.setdefault("robust_confidence_intervals", {})[policy] = (
                            float(robust_ci[0]),
                            float(robust_ci[1]),
                        )

            # Extract metadata if available (filtering out verbose fields)
            if hasattr(estimation_result, "metadata") and estimation_result.metadata:
                # Filter out verbose/internal fields that bloat results
                filtered_metadata = {
                    k: v
                    for k, v in estimation_result.metadata.items()
                    if k
                    not in [
                        "if_sample_indices"
                    ]  # This is just [0, 1, 2, ...] - not needed
                }
                if filtered_metadata:  # Only store if non-empty after filtering
                    result["metadata"] = filtered_metadata

            # Extract DR-specific diagnostics if available
            if (
                hasattr(estimation_result, "diagnostics")
                and estimation_result.diagnostics
            ):
                diag = estimation_result.diagnostics

                # Extract overall status
                if hasattr(diag, "overall_status"):
                    result["overall_status"] = str(
                        diag.overall_status.name
                        if hasattr(diag.overall_status, "name")
                        else diag.overall_status
                    )

                # Extract outcome model R² for DR estimators
                if hasattr(diag, "outcome_r2_range"):
                    result["outcome_r2_min"], result["outcome_r2_max"] = (
                        diag.outcome_r2_range
                    )

                # Check for DR diagnostics
                if (
                    hasattr(diag, "dr_diagnostics_per_policy")
                    and diag.dr_diagnostics_per_policy is not None
                ):
                    for policy, policy_diag in diag.dr_diagnostics_per_policy.items():
                        if isinstance(policy_diag, dict):
                            # Extract outcome model R² per policy
                            if "r2_oof" in policy_diag:
                                result.setdefault("outcome_r2", {})[policy] = (
                                    policy_diag["r2_oof"]
                                )
                            if "orthogonality_score" in policy_diag:
                                result.setdefault("orthogonality_score", {})[policy] = (
                                    policy_diag["orthogonality_score"]
                                )
                            if "mc_variance_share" in policy_diag:
                                result.setdefault("mc_variance_share", {})[policy] = (
                                    policy_diag["mc_variance_share"]
                                )
                            if "draws_per_prompt" in policy_diag:
                                result["draws_per_prompt"] = policy_diag[
                                    "draws_per_prompt"
                                ]
                # Also check for orthogonality_scores directly
                if hasattr(diag, "orthogonality_scores"):
                    result["orthogonality_scores"] = diag.orthogonality_scores
                # Check for IIC diagnostics
                if hasattr(diag, "iic_diagnostics") and diag.iic_diagnostics:
                    result["iic_diagnostics"] = diag.iic_diagnostics

            # Also check metadata for IIC diagnostics (DR estimators store them there)
            if hasattr(estimation_result, "metadata") and estimation_result.metadata:
                if "iic_diagnostics" in estimation_result.metadata:
                    result["iic_diagnostics"] = estimation_result.metadata[
                        "iic_diagnostics"
                    ]
                # OC-DR-CPO stores orthogonality scores in metadata
                if "orthogonality_scores" in estimation_result.metadata:
                    result["orthogonality_scores"] = estimation_result.metadata[
                        "orthogonality_scores"
                    ]
                # Extract MC variance diagnostics from metadata
                if "mc_variance_diagnostics" in estimation_result.metadata:
                    mc_diag = estimation_result.metadata["mc_variance_diagnostics"]
                    # Store full MC diagnostics structure
                    result["mc_diagnostics"] = {}
                    all_m_mins = []
                    all_m_maxs = []
                    for policy, diag in mc_diag.items():
                        if isinstance(diag, dict):
                            result["mc_diagnostics"][policy] = {
                                "M_min": diag.get(
                                    "min_draws_per_prompt", diag.get("M_min")
                                ),
                                "M_max": diag.get(
                                    "max_draws_per_prompt", diag.get("M_max")
                                ),
                                "mc_var_fraction": diag.get(
                                    "mc_share", diag.get("mc_variance_share")
                                ),
                            }
                            # Collect for aggregation
                            m_min = diag.get("min_draws_per_prompt", diag.get("M_min"))
                            m_max = diag.get("max_draws_per_prompt", diag.get("M_max"))
                            if m_min is not None:
                                all_m_mins.append(m_min)
                            if m_max is not None:
                                all_m_maxs.append(m_max)

                    # Add aggregated values at top level for easy access
                    if all_m_mins:
                        result["mc_diagnostics"]["M_min"] = min(all_m_mins)
                    if all_m_maxs:
                        result["mc_diagnostics"]["M_max"] = max(all_m_maxs)

            # Compute diagnostics
            self.compute_diagnostics(estimator, result, len(dataset.samples))

            # Restore oracle labels for ground truth computation (top-level field)
            if original_oracle_labels:
                for idx, oracle_label in original_oracle_labels.items():
                    dataset.samples[idx].oracle_label = oracle_label

            # Compute oracle truths
            oracle_truths = self._load_oracle_ground_truth(
                spec.dataset_path,
                dataset,
                list(sampler.target_policies),
            )
            result["oracle_truths"] = oracle_truths
            # If available, attach per-policy oracle counts to result
            if hasattr(self, "_oracle_counts_per_policy") and isinstance(
                self._oracle_counts_per_policy, dict
            ):
                result["n_oracle_per_policy"] = {
                    k: int(v) for k, v in self._oracle_counts_per_policy.items()
                }

            # Compute RMSE (excluding unhelpful policy which has different distribution)
            result["rmse_vs_oracle"] = self._compute_rmse(
                result["estimates"], oracle_truths
            )

            # Mean CI width
            if result["confidence_intervals"]:
                widths = [
                    ci[1] - ci[0] for ci in result["confidence_intervals"].values()
                ]
                result["mean_ci_width"] = float(np.mean(widths))

            result["success"] = True

        except Exception as e:
            import traceback
            logger.error(f"Experiment failed: {e}")
            logger.error(traceback.format_exc())
            result["error"] = str(e)
            result["success"] = False

        result["runtime_s"] = time.time() - result["start_ts"]

        # Convert numpy bools in diagnostic outputs to Python bools for JSON serialization
        # These come from the CJE library diagnostics
        if (
            "orthogonality_scores" in result
            and result["orthogonality_scores"] is not None
        ):
            for policy, scores in result["orthogonality_scores"].items():
                if isinstance(scores, dict) and "passes_test" in scores:
                    scores["passes_test"] = bool(scores["passes_test"])

        if "iic_diagnostics" in result and result["iic_diagnostics"] is not None:
            for policy, diag in result["iic_diagnostics"].items():
                if isinstance(diag, dict) and "mean_preserved" in diag:
                    diag["mean_preserved"] = bool(diag["mean_preserved"])

        # Convert all numpy types to Python types for JSON serialization
        result = self._convert_numpy(result)
        return result

    def _convert_numpy(self, obj: Any) -> Any:
        """Convert numpy types to Python types for JSON serialization."""
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy(v) for v in obj]
        elif hasattr(obj, "item"):
            return obj.item()
        return obj

    def run_with_seeds(self, spec: ExperimentSpec) -> List[Dict[str, Any]]:
        """Run experiment with multiple seeds.

        Args:
            spec: Experiment specification

        Returns:
            List of results (one per seed)
        """
        results = []
        for i in range(spec.n_seeds):
            seed = spec.seed_base + i
            logger.info(f"Running {self.name} with seed {seed} ({i+1}/{spec.n_seeds})")
            result = self.run_single(spec, seed)
            results.append(result)

            # Log progress
            if result["success"]:
                logger.info(f"  ✓ RMSE: {result.get('rmse_vs_oracle', 'N/A'):.4f}")
            else:
                logger.warning(f"  ✗ Failed: {result.get('error', 'Unknown')}")

        return results

    def run_ablation(self) -> List[Dict[str, Any]]:
        """Run the complete ablation.

        Override this in subclasses to define the experiment grid.
        """
        raise NotImplementedError("Subclasses must implement run_ablation()")
