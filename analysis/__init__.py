"""CJE analysis pipeline modules.

This package contains modular components for analyzing CJE datasets.
Each module has a single responsibility following the Unix philosophy.
"""

from .loading import load_data
from .calibration import handle_rewards, restore_oracle_labels
from .estimation import create_estimator, add_fresh_draws
from .results import display_results, compute_base_statistics
from .diagnostics import (
    display_weight_diagnostics,
    display_dr_diagnostics,
    display_augmentation_diagnostics,
    analyze_extreme_weights_report,
)
from .visualization import generate_visualizations
from .export import export_results

__all__ = [
    # Data loading
    "load_data",
    # Calibration
    "handle_rewards",
    "restore_oracle_labels",
    # Estimation
    "create_estimator",
    "add_fresh_draws",
    # Results
    "display_results",
    "compute_base_statistics",
    # Diagnostics
    "display_weight_diagnostics",
    "display_dr_diagnostics",
    "display_augmentation_diagnostics",
    "analyze_extreme_weights_report",
    # Visualization
    "generate_visualizations",
    # Export
    "export_results",
]
