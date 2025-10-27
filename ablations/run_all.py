#!/usr/bin/env python3
"""Complete CJE ablation analysis pipeline.

This script runs all ablations and generates all paper artifacts:
1. Runs unified experiments (parameter sweeps)
2. Generates tables for the paper
3. Generates all visualizations

Usage:
    cd ablations
    python run_all.py
"""

import sys
import logging
from pathlib import Path

# Add parent to path for cje imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    """Run complete analysis pipeline."""
    print("=" * 70)
    print("CJE COMPLETE ABLATION ANALYSIS")
    print("=" * 70)

    # 1. Run experiments
    print("\n[Step 1/3] Running ablation experiments...")
    print("-" * 50)
    from run import UnifiedAblation

    ablation = UnifiedAblation()
    results = ablation.run_ablation()

    print(f"\n✓ Completed {len(results)} experiments")

    # 2. Generate tables and basic analysis
    print("\n[Step 2/3] Generating analysis tables...")
    print("-" * 50)
    from analyze import main as analyze_main

    # Run analysis
    analyze_main()

    print("\n✓ Generated tables in ablations/results/analysis/")

    # 3. Generate comprehensive visualizations
    print("\n[Step 3/3] Generating paper visualizations...")
    print("-" * 50)

    try:
        # Try to run legacy visualization code if it exists
        from estimator_comparison import EstimatorComparisonAblation

        print("Creating estimator comparison plots...")
        ec = EstimatorComparisonAblation()
        results_path = Path("results/estimator_comparison")
        results_path.mkdir(parents=True, exist_ok=True)

        # Note: These would need the results in the format they expect
        # For now, we'll skip if they don't work
        try:
            ec.create_figure()
            print("✓ Created main comparison figure")
        except Exception as e:
            logger.warning(f"Could not create comparison figure: {e}")

        try:
            ec.create_all_scenario_plots()
            print("✓ Created policy heterogeneity plots")
        except Exception as e:
            logger.warning(f"Could not create heterogeneity plots: {e}")

    except ImportError:
        logger.info("Legacy visualization code not available (estimator_comparison)")

    try:
        from sample_size import SampleSizeAblation

        print("Creating sample size scaling plots...")
        ss = SampleSizeAblation()
        try:
            ss.create_figure()
            print("✓ Created sample scaling figure")
        except Exception as e:
            logger.warning(f"Could not create scaling figure: {e}")

    except ImportError:
        logger.info("Legacy visualization code not available (sample_size)")

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated artifacts:")
    print("  Tables:  results/analysis/*.csv")
    print("  Plots:   results/analysis/*.png")
    print("  Results: results/all_experiments.jsonl")

    return 0


if __name__ == "__main__":
    sys.exit(main())
