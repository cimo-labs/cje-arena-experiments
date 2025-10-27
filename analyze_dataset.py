#!/usr/bin/env python3
"""CJE Analysis Pipeline - Clean Orchestrator.

This is a thin orchestrator that coordinates the analysis pipeline modules.
Each module does one thing well, following the Unix philosophy.

Usage:
    python analyze_refactored.py --data path/to/dataset.jsonl --estimator calibrated-ips
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# CJE imports
from cje.data.precomputed_sampler import PrecomputedSampler

# Analysis pipeline modules
from analysis import (
    load_data,
    handle_rewards,
    restore_oracle_labels,
    create_estimator,
    display_results,
    display_weight_diagnostics,
    display_dr_diagnostics,
    display_augmentation_diagnostics,
    analyze_extreme_weights_report,
    generate_visualizations,
    export_results,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run CJE analysis on Arena data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data",
        default="data/cje_dataset.jsonl",
        help="Path to CJE dataset",
    )

    # Estimator arguments
    parser.add_argument(
        "--estimator",
        default="calibrated-ips",
        choices=[
            "calibrated-ips",
            "raw-ips",
            "dr-cpo",
            "mrdr",
            "tmle",
            "stacked-dr",
        ],
        help="Estimator to use",
    )
    parser.add_argument(
        "--estimator-config",
        type=json.loads,
        help="JSON config for estimator",
    )

    # Oracle/calibration arguments
    parser.add_argument(
        "--use-oracle",
        action="store_true",
        help="Use oracle labels directly as rewards",
    )
    parser.add_argument(
        "--oracle-coverage",
        type=float,
        default=1.0,
        help="Fraction of oracle labels to use for calibration",
    )
    parser.add_argument(
        "--judge-field",
        default="judge_score",
        help="Field name for judge scores",
    )
    parser.add_argument(
        "--oracle-field",
        default="oracle_label",
        help="Field name for oracle labels",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-fitting folds",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for results (JSON or CSV)",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        help="Directory for saving plots",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation",
    )

    # Analysis arguments
    parser.add_argument(
        "--extreme-threshold-low",
        type=float,
        default=1e-10,
        help="Threshold for near-zero weights",
    )
    parser.add_argument(
        "--extreme-threshold-high",
        type=float,
        default=10.0,
        help="Threshold for extreme high weights",
    )

    # Debug arguments
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    return parser.parse_args()


def setup_logging(args: argparse.Namespace) -> None:
    """Configure logging based on arguments."""
    if args.debug:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main() -> int:
    """Run the CJE analysis pipeline.

    This orchestrator coordinates the pipeline modules:
    1. Load data
    2. Handle rewards/calibration
    3. Create and fit estimator
    4. Run estimation
    5. Display results
    6. Display diagnostics
    7. Generate visualizations
    8. Export results

    Returns:
        0 on success, 1 on failure
    """
    # Parse arguments
    args = parse_arguments()
    setup_logging(args)

    # Print header
    if not args.quiet:
        print("\n" + "=" * 50)
        print(f"CJE Analysis Pipeline")
        print(f"Estimator: {args.estimator}")
        print(f"Dataset: {args.data}")
        print("=" * 50)

    try:
        # 1. Load data
        dataset = load_data(args.data, verbose=not args.quiet)

        # 2. Handle rewards and calibration
        analysis_config = {
            "n_folds": args.n_folds,
            "oracle_coverage": args.oracle_coverage,
        }
        calibrated_dataset, cal_result = handle_rewards(
            dataset, args, analysis_config, verbose=not args.quiet
        )

        # 3. Create sampler and estimator
        if not args.quiet:
            print("\n3. Setting up estimator...")
        sampler = PrecomputedSampler(calibrated_dataset)
        estimator = create_estimator(args, sampler, calibrated_dataset, cal_result)

        # 4. Fit and run estimation
        if not args.quiet:
            print(f"   ✓ Created {args.estimator} estimator")
            print(f"   Fitting estimator...")
        estimator.fit()

        if not args.quiet:
            print(f"   Running estimation...")
        results = estimator.estimate()

        if not args.quiet:
            print(f"   ✓ Estimation complete")

        # 5. Restore oracle labels for visualization
        # (They were masked during calibration for partial coverage)
        restore_oracle_labels(calibrated_dataset, args)
        # Also restore on original dataset for oracle comparison
        restore_oracle_labels(dataset, args)

        # 6. Display results
        summary_data = display_results(
            results,
            calibrated_dataset,
            sampler,
            estimator,
            args,
            dataset,
        )

        # 7. Display diagnostics
        weight_diagnostics = display_weight_diagnostics(
            estimator, sampler, calibrated_dataset, args
        )

        # Display DR diagnostics if applicable
        if args.estimator in ["dr-cpo", "mrdr", "tmle"]:
            display_dr_diagnostics(results, args)

        # Display augmentation diagnostics (for all estimators)
        display_augmentation_diagnostics(estimator, results, args.oracle_coverage, args)

        # Analyze extreme weights if requested
        if hasattr(estimator, "get_raw_weights"):
            analyze_extreme_weights_report(estimator, sampler, calibrated_dataset, args)

        # 8. Generate visualizations
        generate_visualizations(
            results,
            dataset,
            calibrated_dataset,
            estimator,
            sampler,
            args,
            summary_data,
            cal_result,
        )

        # 9. Export results
        export_results(
            results,
            dataset,
            summary_data,
            weight_diagnostics,
            args,
        )

        # Success message
        if not args.quiet:
            steps_completed = 7  # Base steps
            if args.estimator in ["dr-cpo", "mrdr", "tmle"]:
                steps_completed += 1  # DR diagnostics
            if not args.no_plots:
                steps_completed += 1  # Visualizations

            print(f"\n✓ Analysis complete! ({steps_completed} steps)")

        return 0

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
