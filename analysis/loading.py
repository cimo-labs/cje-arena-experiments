"""Data loading and validation for CJE analysis.

This module handles loading datasets and basic validation.
Following CLAUDE.md: Do one thing well - this only loads data.
"""

from pathlib import Path
from typing import Any
from cje import load_dataset_from_jsonl


def load_data(data_path: str, verbose: bool = True) -> Any:
    """Load dataset from JSONL file.

    Args:
        data_path: Path to dataset file
        verbose: Whether to print loading status

    Returns:
        Loaded dataset

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If dataset is invalid
    """
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    if verbose:
        print("\n1. Loading dataset...")

    dataset = load_dataset_from_jsonl(data_path)

    if verbose:
        print(f"   ✓ Loaded {dataset.n_samples} samples")
        print(f"   ✓ Target policies: {dataset.target_policies}")

    # Basic validation
    if dataset.n_samples == 0:
        raise ValueError("Dataset is empty")

    if not dataset.target_policies:
        raise ValueError("No target policies found in dataset")

    return dataset
