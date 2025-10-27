# CJE Ablation Experiments

Systematic ablation studies demonstrating the value of calibrated importance sampling and doubly robust methods for off-policy evaluation.

## Quick Start

```bash
# Run all ablation experiments (360 total)
python run.py       # Runs with checkpoint/resume support

# Generate paper tables and analysis
python -m reporting.cli_generate                      # All tables (main + quadrant)
python -m reporting.cli_generate --tables m1,m2,m3    # Main tables only
python -m reporting.cli_generate --format markdown    # Markdown format for quick viewing
```

## Table Generation

Tables are generated using the unified reporting module:

```bash
# Generate all tables (main + quadrant)
python -m reporting.cli_generate --results results/all_experiments.jsonl --output-dir tables/

# Generate specific tables only
python -m reporting.cli_generate --tables m1,m2,m3  # Main tables only
python -m reporting.cli_generate --tables quadrant  # Quadrant tables only
```

Tables are saved to `tables/main/` and `tables/quadrant/`.

## Current System Structure

```
ablations/
├── run.py                     # Main experiment runner with checkpoint/resume
├── run_all.py                 # Batch runner for all experiments
├── config.py                  # Experiment configuration
├── reporting/                 # Table generation and analysis module
│   ├── cli_generate.py       # CLI for generating tables
│   ├── tables_main.py        # Main table builders (M1, M2, M3, etc.)
│   ├── format_latex.py       # LaTeX formatting
│   ├── io.py                 # Data loading and processing
│   ├── metrics.py            # Metric calculations
│   └── aggregate.py          # Aggregation utilities
├── core/                      # Infrastructure
│   ├── base.py               # BaseAblation class
│   └── schemas.py            # Data schemas
└── results/                   # All outputs
    ├── all_experiments.jsonl # Raw experiment results
    ├── checkpoint.jsonl      # Progress tracking for resume
    └── tables/               # Generated tables (via reporting module)
        ├── main/            # Main paper tables
        └── quadrant/        # Quadrant-specific tables
```

## What Gets Tested

The unified system (`run.py`) tests all combinations of:

### Parameters
- **Estimators**: raw-ips, calibrated-ips, orthogonalized-ips, dr-cpo, oc-dr-cpo, tr-cpo, stacked-dr
- **Sample sizes**: 500, 1000, 2500, 5000
- **Oracle coverage**: 5%, 10%, 25%, 50%, 100%
- **Weight Calibration (SIMCal)**: On/off (controlled by `use_weight_calibration`, with constraints)
- **Covariates**: On/off (controlled by `use_covariates`) - tests with/without response_length covariate
- **IIC (Isotonic Influence Control)**: On/off for all methods (controlled by `use_iic`)
- **Seed**: Single seed (42) for reproducibility

### Estimator Constraints
Some estimators have calibration requirements:
- **Always calibrated**: calibrated-ips, orthogonalized-ips, oc-dr-cpo
- **Never calibrated**: raw-ips, tr-cpo (TR-CPO uses raw/Hájek weights for theoretical correctness)
- **Optional calibration**: dr-cpo, stacked-dr

This reduces the total experiments to 3,600 valid combinations (10 seeds × 360 unique configs).

### Expected Results
- **SIMCal weight calibration**: Should improve DR methods (reduces weight variance)
- **Covariates**: Response-level features (response_length) can improve calibration and outcome modeling
- **IIC**: 5-20% standard error reduction (variance-only, preserves point estimates)
- **DR methods**: Critical when n < 500 or oracle < 10%
- **Stacked-DR**: Most robust across scenarios

## Output Files

### Raw Results
- `results/all_experiments.jsonl`: One line per experiment with:
  - `spec`: Configuration (estimator, sample_size, oracle_coverage, etc.)
  - `estimates`: Policy value estimates
  - `standard_errors`: Uncertainty estimates
  - `rmse_vs_oracle`: Error vs ground truth
  - `orthogonality_scores`: DR orthogonality diagnostic (should be near 0)
  - `iic_diagnostics`: Per-policy R² and SE reduction from IIC
  - `ess_relative`: Effective sample size as percentage
  - `hellinger_affinity`: Structural overlap measure (higher is better)

### Analysis Outputs
- `results/analysis/main_comparison.csv`: Summary table
- `results/analysis/calibration_comparison.csv`: Calibration on/off comparison
- `results/analysis/iic_comparison.csv`: IIC on/off comparison
- `results/analysis/rmse_by_configuration.png`: Main visualization

## Running Specific Configurations

To test specific settings, modify `config.py`:

```python
EXPERIMENTS = {
    'estimators': ['calibrated-ips', 'dr-cpo'],  # Just two
    'sample_sizes': [1000],                       # Single size
    'oracle_coverages': [0.10],                   # Single coverage
    'use_weight_calibration': [True],             # Just with weight calibration
    'use_iic': [False],                          # No IIC
    'n_seeds': 2,                                 # Fewer seeds
}
```

Then run: `python run.py`

## Legacy Code

The `legacy/` directory contains the original ablation implementation for reference. This code has been fully replaced by the unified system but is preserved for validation and historical context.

## Notes

- All diagnostics now come from the main CJE library (no local duplicates)
- The system uses the Arena 10k dataset by default (see config.py to change)
- Fresh draws are auto-loaded for DR methods when available