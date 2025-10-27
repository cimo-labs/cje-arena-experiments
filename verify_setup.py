#!/usr/bin/env python3
"""Verify that the CJE Arena Experiments environment is set up correctly."""

import sys
from pathlib import Path


def check_import(module_name: str, package_name: str = None) -> bool:
    """Check if a module can be imported."""
    display_name = package_name or module_name
    try:
        __import__(module_name)
        print(f"✓ {display_name} installed")
        return True
    except ImportError:
        print(f"✗ {display_name} not found - install with: pip install {package_name or module_name}")
        return False


def check_file(filepath: Path, required: bool = True) -> bool:
    """Check if a file exists."""
    exists = filepath.exists()
    size_str = f"({filepath.stat().st_size / 1024 / 1024:.1f}MB)" if exists else ""
    marker = "✓" if exists else ("✗" if required else "⚠")
    status = "found" if exists else ("MISSING" if required else "optional, not found")
    print(f"{marker} {filepath} {size_str} - {status}")
    return exists or not required


def main() -> int:
    """Run all verification checks."""
    print("=" * 60)
    print("CJE Arena Experiments - Setup Verification")
    print("=" * 60)

    all_passed = True

    # Check Python version
    print("\n1. Python Version:")
    py_version = sys.version_info
    if py_version >= (3, 9) and py_version < (3, 13):
        print(f"✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        print(f"✗ Python {py_version.major}.{py_version.minor} - requires 3.9-3.12")
        all_passed = False

    # Check required packages
    print("\n2. Required Packages:")
    all_passed &= check_import("cje", "cje-eval")
    all_passed &= check_import("numpy")
    all_passed &= check_import("pandas")
    all_passed &= check_import("scipy")
    all_passed &= check_import("sklearn", "scikit-learn")
    all_passed &= check_import("matplotlib")

    # Check CJE version
    try:
        import cje
        version = getattr(cje, "__version__", "unknown")
        print(f"  CJE version: {version}")
        if version != "unknown":
            major, minor, patch = map(int, version.split('.'))
            if (major, minor, patch) >= (0, 2, 3):
                print(f"  ✓ Version {version} meets minimum requirement (0.2.3)")
            else:
                print(f"  ⚠ Version {version} is older than 0.2.3, consider upgrading")
    except (ImportError, ValueError, AttributeError):
        pass

    # Check data files
    print("\n3. Data Files:")
    data_dir = Path("data")
    all_passed &= check_file(data_dir / "cje_dataset.jsonl", required=True)
    all_passed &= check_file(data_dir / "prompts.jsonl", required=True)
    check_file(data_dir / "responses", required=False)
    check_file(data_dir / "logprobs", required=False)

    # Try to load and parse dataset
    print("\n4. Dataset Loading:")
    try:
        from cje import load_dataset_from_jsonl
        dataset = load_dataset_from_jsonl("data/cje_dataset.jsonl")
        print(f"✓ Successfully loaded {len(dataset.samples)} samples")

        # Check for required fields
        sample = dataset.samples[0]
        has_judge = hasattr(sample.metadata, 'judge_score')
        has_oracle = hasattr(sample.metadata, 'oracle_label')
        print(f"  - Judge scores: {'✓' if has_judge else '✗'}")
        print(f"  - Oracle labels: {'✓' if has_oracle else '✗'}")

        if not (has_judge and has_oracle):
            print("  ⚠ Dataset missing required metadata fields")
            all_passed = False

    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All checks passed! You're ready to run experiments.")
        print("\nTry: python analyze_dataset.py --data data/cje_dataset.jsonl")
        return 0
    else:
        print("✗ Some checks failed. Please address the issues above.")
        print("\nInstall missing dependencies with: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
