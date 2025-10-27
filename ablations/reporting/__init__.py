"""
Reporting module for CJE ablation experiments (v2.0).

This module provides a clean, modular pipeline for generating journal-quality
tables from experiment results. The design uses regime-based analysis matrices
instead of competition-style leaderboards.

Structure:
- io.py: Tidy data loading and deduplication
- metrics.py: Pure metric functions over tidy DataFrames
- aggregate.py: Groupby wrappers and paired delta computation
- tables_main.py: Main table builders (M1-M3)
- format_latex.py: LaTeX formatting utilities
- cli_generate.py: Unified CLI for table generation

Usage:
    python -m reporting.cli_generate \
        --results results/all_experiments.jsonl \
        --output tables/ \
        --format latex
"""

from . import io
from . import metrics
from . import aggregate
from . import tables_main
from . import format_latex

__all__ = [
    "io",
    "metrics",
    "aggregate",
    "tables_main",
    "format_latex",
]

# Version
__version__ = "2.1.0"
