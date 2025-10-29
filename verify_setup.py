#!/usr/bin/env python3
"""
Quick verification script to check that the CJE arena experiments are set up correctly.

Run this after installation to verify:
1. CJE library is installed
2. Data files are present
3. A minimal analysis runs successfully
"""

import sys
from pathlib import Path


def check_cje_install():
    """Verify CJE library is installed."""
    print("Checking CJE installation...", end=" ")
    try:
        import cje
        print(f"✓ (version {cje.__version__})")
        return True
    except ImportError:
        print("✗")
        print("  ERROR: CJE not installed. Run: pip install -r requirements.txt")
        return False


def check_data_files():
    """Verify required data files exist."""
    print("Checking data files...", end=" ")
    data_dir = Path("data")

    if not data_dir.exists():
        print("✗")
        print("  ERROR: data/ directory not found")
        return False

    required_files = [
        "data/cje_dataset.jsonl",
        "data/prompts.jsonl",
    ]

    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)

    if missing:
        print("✗")
        print(f"  ERROR: Missing files: {', '.join(missing)}")
        return False

    print("✓")
    return True


def check_fresh_draws():
    """Check if fresh draws directory exists."""
    print("Checking fresh draws...", end=" ")
    fresh_draws = Path("data/responses")

    if not fresh_draws.exists():
        print("⚠")
        print("  WARNING: data/responses/ not found (needed for DR estimators)")
        print("  Direct methods will work, but DR/TMLE/MRDR require fresh draws")
        return True

    # Count response files
    response_files = list(fresh_draws.glob("*_responses.jsonl"))
    if len(response_files) < 3:
        print("⚠")
        print(f"  WARNING: Only {len(response_files)} response files found")
        return True

    print(f"✓ ({len(response_files)} policies)")
    return True


def run_minimal_test():
    """Run a minimal CJE analysis to verify everything works."""
    print("\nRunning minimal analysis test...", end=" ")

    try:
        from cje import load_dataset_from_jsonl
        from cje.calibration import calibrate_dataset

        # Load small sample
        dataset = load_dataset_from_jsonl("data/cje_dataset.jsonl", max_samples=100)

        # Try calibration
        calibrated = calibrate_dataset(
            dataset,
            oracle_coverage=0.25,
            calibration_mode="auto",
            random_seed=42
        )

        print("✓")
        print(f"  Successfully calibrated {len(calibrated.samples)} samples")
        return True

    except Exception as e:
        print("✗")
        print(f"  ERROR: {str(e)}")
        return False


def main():
    print("=" * 60)
    print("CJE Arena Experiments - Setup Verification")
    print("=" * 60)
    print()

    checks = [
        check_cje_install(),
        check_data_files(),
        check_fresh_draws(),
        run_minimal_test(),
    ]

    print()
    print("=" * 60)

    if all(checks):
        print("✓ All checks passed! Setup is ready.")
        print()
        print("Next steps:")
        print("  • Run a quick analysis: python analyze_dataset.py")
        print("  • Run ablations: cd ablations && python run.py")
        print("  • See README.md for full documentation")
        return 0
    else:
        print("✗ Some checks failed. See errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
