#!/usr/bin/env python3
"""
Quick verification script to check that the CJE arena experiments are set up correctly.

Run this after installation to verify:
1. CJE library is installed
2. Data files are present (included in repo)
3. A minimal analysis runs successfully
"""

import sys
from pathlib import Path


def check_cje_install():
    """Verify CJE library is installed."""
    print("Checking CJE installation...", end=" ")
    try:
        import cje
        print(f"OK (version {cje.__version__})")
        return True
    except ImportError:
        print("MISSING")
        print("  Run: pip install -r requirements.txt")
        return False


def check_data_files():
    """Verify required data files exist."""
    print("Checking data files...", end=" ")
    data_dir = Path("data")

    if not data_dir.exists():
        print("MISSING")
        print("  Data directory not found - repo may be incomplete")
        return False

    required_files = [
        "data/cje_dataset.jsonl",
    ]

    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)

    if missing:
        print("MISSING")
        print(f"  Missing: {', '.join(missing)}")
        print("  Data files should be included in repo - check git clone")
        return False

    # Check file size to ensure it's not empty/corrupted
    dataset_size = Path("data/cje_dataset.jsonl").stat().st_size
    if dataset_size < 1_000_000:  # Should be ~10MB
        print("CORRUPTED")
        print(f"  File too small ({dataset_size} bytes)")
        print("  Re-clone the repository to get fresh data")
        return False

    print("OK")
    return True


def check_fresh_draws():
    """Check if fresh draws directory exists."""
    print("Checking fresh draws...", end=" ")
    fresh_draws = Path("data/responses")

    if not fresh_draws.exists():
        print("MISSING (optional)")
        print("  Fresh draws needed for DR estimators")
        print("  Re-clone repo or regenerate with data_generation scripts")
        return True  # Not a failure, just a warning

    # Count response files
    response_files = list(fresh_draws.glob("*_responses.jsonl"))
    if len(response_files) < 3:
        print(f"PARTIAL ({len(response_files)} policies)")
        print("  Some fresh draw files missing")
        return True

    print(f"OK ({len(response_files)} policies)")
    return True


def check_results():
    """Check if pre-computed results exist."""
    print("Checking pre-computed results...", end=" ")
    results_file = Path("ablations/results/all_experiments.jsonl")

    if not results_file.exists():
        print("NOT FOUND (optional)")
        print("  Pre-computed results not found (1.3GB)")
        print("  To regenerate: cd ablations && python run.py")
        return True  # Not a failure - results are optional

    # Check size
    size_gb = results_file.stat().st_size / (1024**3)
    print(f"OK ({size_gb:.1f}GB)")
    return True


def run_minimal_test():
    """Run a minimal CJE analysis to verify everything works."""
    print("\nRunning minimal analysis test...", end=" ")

    try:
        from cje import load_dataset_from_jsonl
        from cje.calibration import calibrate_dataset

        # Load dataset (full load, but calibration is fast)
        dataset = load_dataset_from_jsonl("data/cje_dataset.jsonl")

        # Try calibration
        calibrated, _ = calibrate_dataset(
            dataset,
            oracle_field="oracle_label",
            judge_field="judge_score",
            calibration_mode="auto",
            random_seed=42,
        )

        print("OK")
        print(f"  Calibrated {len(calibrated.samples)} samples successfully")
        return True

    except Exception as e:
        print("FAILED")
        print(f"  ERROR: {str(e)}")
        return False


def main():
    print("=" * 60)
    print("CJE Arena Experiments - Setup Verification")
    print("=" * 60)
    print()

    checks = [
        ("CJE library", check_cje_install()),
        ("Data files", check_data_files()),
        ("Fresh draws", check_fresh_draws()),
        ("Results", check_results()),
    ]

    # Only run minimal test if data is present
    if checks[1][1]:  # Data files check passed
        checks.append(("Analysis test", run_minimal_test()))

    print()
    print("=" * 60)

    required_passed = all(passed for name, passed in checks[:2])  # CJE + data
    all_passed = all(passed for name, passed in checks)

    if required_passed:
        print("Setup is ready!")
        print()
        print("Quick start:")
        print("  python analyze_dataset.py --data data/cje_dataset.jsonl")
        print()
        print("Run ablations:")
        print("  cd ablations && python run.py")
        return 0
    else:
        print("Setup incomplete. See errors above.")
        print()
        print("To fix:")
        print("  1. pip install -r requirements.txt")
        print("  2. Ensure repo was cloned with all data files")
        return 1


if __name__ == "__main__":
    sys.exit(main())
