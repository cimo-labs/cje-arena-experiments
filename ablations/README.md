# CJE Ablation Experiments

Systematic ablation studies demonstrating the value of calibrated importance sampling and doubly robust methods for off-policy evaluation.

## Quick Start

```bash
# Run all ablation experiments (~19,000 total across 50 seeds)
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
- **Estimators**: direct, naive-direct, raw-ips, calibrated-ips, dr-cpo, tr-cpo-e, stacked-dr
- **Sample sizes**: 250, 500, 1000, 2500, 5000
- **Oracle coverage**: 5%, 10%, 25%, 50%, 100%
- **Weight Calibration (SIMCal)**: On/off (controlled by `use_weight_calibration`, with constraints)
- **Covariates**: On/off (controlled by `use_covariates`) - tests with/without response_length covariate
- **Seeds**: 50 seeds (0-49) for robust variance estimates

### Estimator Constraints
Some estimators have calibration requirements:
- **Always calibrated**: calibrated-ips, stacked-dr
- **Never calibrated**: direct, naive-direct, raw-ips, tr-cpo-e
- **Optional calibration**: dr-cpo

Total: ~19,000 experiments across all parameter combinations.

### Expected Results
- **SIMCal weight calibration**: Improves ESS from <1% to >80% (see paper Table 3)
- **Covariates**: Response-level features (response_length) improve calibration
- **Direct+cov**: Best ranking accuracy (94.3% pairwise)
- **DR methods**: Robust under limited overlap
- **Stacked-DR**: Most robust across scenarios

## Output Files

### Raw Results
- `results/all_experiments.jsonl`: One line per experiment with:
  - `spec`: Configuration (estimator, sample_size, oracle_coverage, etc.)
  - `estimates`: Policy value estimates
  - `standard_errors`: Uncertainty estimates
  - `rmse_vs_oracle`: Error vs ground truth
  - `ess_relative`: Effective sample size as percentage
  - `hellinger_affinity`: Structural overlap measure (higher is better)

### Generated Tables
- `tables/main/`: Main paper tables (M1 accuracy, M2 ESS comparison, M3 gates)
- `tables/quadrant/`: Quadrant-specific breakdowns

## Running Specific Configurations

To test specific settings, modify `config.py`:

```python
EXPERIMENTS = {
    'estimators': ['calibrated-ips', 'dr-cpo'],  # Just two
    'sample_sizes': [1000],                       # Single size
    'oracle_coverages': [0.10],                   # Single coverage
    'use_weight_calibration': [True],             # Just with weight calibration
    'use_covariates': [False],                    # Without covariates
    'seeds': np.arange(0, 5, 1),                  # Fewer seeds
}
```

Then run: `python run.py`

## Notes

- All diagnostics come from the CJE library (no local duplicates)
- Fresh draws are auto-loaded for DR methods when available
- Checkpoint/resume support: experiments resume from where they left off