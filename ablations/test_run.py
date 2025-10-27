#!/usr/bin/env python3
"""
Minimal test to verify ablations can run from scratch.

Tests a single configuration:
- 1 estimator (direct)
- 1 sample size (250)
- 1 oracle coverage (0.10)
- 1 seed (42)
"""

import sys
import logging
from pathlib import Path

# Add parent directories to path (same as run.py)
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from config import DATA_PATH
from core.base import BaseAblation
from core.schemas import ExperimentSpec

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    """Run minimal test configuration."""
    print("=" * 70)
    print("MINIMAL ABLATION TEST")
    print("=" * 70)
    print()

    # Check data exists
    if not Path(DATA_PATH).exists():
        print(f"✗ Data file not found: {DATA_PATH}")
        print("  Please ensure the arena dataset is available")
        return 1

    print(f"✓ Data file exists: {DATA_PATH}")
    print()

    # Create minimal test configurations
    test_configs = [
        ("direct", False),
        ("raw-ips", False),
        ("calibrated-ips", True),
        ("dr-cpo", True),
    ]

    print(f"Testing {len(test_configs)} estimator configurations:")
    for estimator, use_cal in test_configs:
        cal_str = "with calibration" if use_cal else "without calibration"
        print(f"  - {estimator} ({cal_str})")
    print()

    ablation = BaseAblation("test")
    results = []

    for estimator, use_weight_calibration in test_configs:
        print(f"\nTesting {estimator}...")
        print("-" * 50)

        spec = ExperimentSpec(
            ablation="test",
            dataset_path=str(DATA_PATH),
            estimator=estimator,
            sample_size=250,
            oracle_coverage=0.10,
            n_seeds=1,
            seed_base=42,
            extra={
                "use_weight_calibration": use_weight_calibration,
                "reward_calibration_mode": "auto",
                "var_cap": 1.0,
            },
        )

        try:
            result = ablation.run_single(spec, seed=42)
            if result["success"]:
                rmse = result.get("rmse_vs_oracle", float("nan"))
                print(f"✓ {estimator} succeeded")
                print(f"  RMSE vs oracle: {rmse:.4f}")
                print(f"  Runtime: {result.get('runtime_s', 0):.2f}s")
                results.append(result)
            else:
                print(f"✗ {estimator} failed: {result.get('error', 'unknown')}")
                return 1
        except Exception as e:
            print(f"✗ {estimator} raised exception: {e}")
            import traceback

            traceback.print_exc()
            return 1

    # Summary
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✓ All {len(results)} configurations ran successfully")
    print()
    print("Ablations are ready to run from scratch!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
