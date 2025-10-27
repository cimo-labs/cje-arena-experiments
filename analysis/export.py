"""Export functionality for CJE analysis results.

This module handles exporting analysis results to various formats (JSON, CSV).

Following CLAUDE.md: Do one thing well - this module only handles export.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def export_results(
    results: Any,
    dataset: Any,
    summary_data: Dict[str, Any],
    weight_diagnostics: Dict[str, Any],
    args: Any,
) -> None:
    """Export results to JSON or CSV format.

    Args:
        results: EstimationResult object
        dataset: Original dataset
        summary_data: Summary statistics from results display
        weight_diagnostics: Weight diagnostic information
        args: Command-line arguments (contains output path)
    """
    if not args.output:
        return

    # Prepare output data
    output_data = _prepare_output_data(
        results, dataset, summary_data, weight_diagnostics, args
    )

    # Determine format from extension
    output_path = Path(args.output)

    if output_path.suffix.lower() == ".csv":
        _export_to_csv(output_data, output_path)
    else:
        # Default to JSON
        _export_to_json(output_data, output_path)

    print(f"\n✓ Results written to: {args.output}")


def _prepare_output_data(
    results: Any,
    dataset: Any,
    summary_data: Dict[str, Any],
    weight_diagnostics: Dict[str, Any],
    args: Any,
) -> Dict[str, Any]:
    """Prepare data structure for export.

    Args:
        results: EstimationResult object
        dataset: Original dataset
        summary_data: Summary statistics
        weight_diagnostics: Weight diagnostics
        args: Command-line arguments

    Returns:
        Dictionary ready for export
    """
    best_policy = summary_data.get("best_policy")
    best_diag = weight_diagnostics.get(best_policy) if best_policy else None

    output_data: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "path": args.data,
            "n_samples": dataset.n_samples,
            "target_policies": dataset.target_policies,
        },
        "estimation": {
            "estimator": args.estimator,
            "estimator_config": args.estimator_config,
            "policies": {},
        },
        "best_policy": best_policy,
        "weight_diagnostics": {},
    }

    # Add weight diagnostics for best policy
    if best_diag and isinstance(best_diag, dict):
        weight_diags: Dict[str, Any] = output_data["weight_diagnostics"]
        weight_diags["best_policy"] = {
            "ess_fraction": float(best_diag.get("ess_fraction", 1.0)),
            "max_weight": float(best_diag.get("max_weight", 1.0)),
            "mean_weight": float(best_diag.get("mean_weight", 1.0)),
        }

    # Add base policy results
    estimation: Dict[str, Any] = output_data["estimation"]
    policies: Dict[str, Any] = estimation["policies"]
    policies["base"] = {
        "estimate": float(summary_data["base_mean"]),
        "standard_error": float(summary_data["base_se"]),
        "ci_lower": float(summary_data["base_ci_lower"]),
        "ci_upper": float(summary_data["base_ci_upper"]),
    }

    # Add target policy results
    ci_lower, ci_upper = results.confidence_interval(alpha=0.05)
    for policy, estimate, se, ci_l, ci_u in zip(
        summary_data["target_policies"],
        results.estimates,
        results.standard_errors,
        ci_lower,
        ci_upper,
    ):
        policies[policy] = {
            "estimate": float(estimate),
            "standard_error": float(se),
            "ci_lower": float(ci_l),
            "ci_upper": float(ci_u),
        }

    # Add diagnostic information if available
    if hasattr(results, "diagnostics") and results.diagnostics:
        output_data["diagnostics"] = _extract_diagnostics(results.diagnostics)

    return output_data


def _export_to_json(output_data: Dict[str, Any], output_path: Path) -> None:
    """Export data to JSON format.

    Args:
        output_data: Data to export
        output_path: Path to output file
    """
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


def _export_to_csv(output_data: Dict[str, Any], output_path: Path) -> None:
    """Export data to CSV format.

    Args:
        output_data: Data to export
        output_path: Path to output file
    """
    import csv

    # Flatten the nested structure for CSV
    rows = []

    # Header row
    header = [
        "policy",
        "estimate",
        "standard_error",
        "ci_lower",
        "ci_upper",
        "is_best",
    ]

    # Add data rows
    best_policy = output_data.get("best_policy")
    for policy, data in output_data["estimation"]["policies"].items():
        row = [
            policy,
            data["estimate"],
            data["standard_error"],
            data["ci_lower"],
            data["ci_upper"],
            "Yes" if policy == best_policy else "No",
        ]
        rows.append(row)

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _extract_diagnostics(diagnostics: Any) -> Dict[str, Any]:
    """Extract key diagnostic information for export.

    Args:
        diagnostics: Diagnostics object (IPSDiagnostics or DRDiagnostics)

    Returns:
        Dictionary of diagnostic information
    """
    diag_dict = {}

    # Common fields
    if hasattr(diagnostics, "n_samples_total"):
        diag_dict["n_samples_total"] = diagnostics.n_samples_total
    if hasattr(diagnostics, "n_samples_valid"):
        diag_dict["n_samples_valid"] = diagnostics.n_samples_valid
    if hasattr(diagnostics, "weight_ess"):
        diag_dict["overall_ess"] = float(diagnostics.weight_ess)

    # IPS-specific fields
    if hasattr(diagnostics, "ess_per_policy"):
        diag_dict["ess_per_policy"] = {
            k: float(v) for k, v in diagnostics.ess_per_policy.items()
        }
    if hasattr(diagnostics, "max_weight_per_policy"):
        diag_dict["max_weight_per_policy"] = {
            k: float(v) for k, v in diagnostics.max_weight_per_policy.items()
        }

    # DR-specific fields
    if hasattr(diagnostics, "dr_cross_fitted"):
        diag_dict["dr_cross_fitted"] = diagnostics.dr_cross_fitted
    if hasattr(diagnostics, "dr_n_folds"):
        diag_dict["dr_n_folds"] = diagnostics.dr_n_folds
    if hasattr(diagnostics, "outcome_r2_range"):
        diag_dict["outcome_r2_range"] = list(diagnostics.outcome_r2_range)
    if hasattr(diagnostics, "worst_if_tail_ratio"):
        diag_dict["worst_if_tail_ratio"] = float(diagnostics.worst_if_tail_ratio)

    # Calibration fields
    if hasattr(diagnostics, "calibration_rmse"):
        diag_dict["calibration_rmse"] = (
            float(diagnostics.calibration_rmse)
            if diagnostics.calibration_rmse is not None
            else None
        )
    if hasattr(diagnostics, "calibration_r2"):
        diag_dict["calibration_r2"] = (
            float(diagnostics.calibration_r2)
            if diagnostics.calibration_r2 is not None
            else None
        )

    return diag_dict
