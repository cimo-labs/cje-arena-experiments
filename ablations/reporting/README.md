# Reporting Module v2.0

This module generates journal-quality tables for CJE ablation experiments, using regime-based analysis matrices instead of competition-style leaderboards.

## Quick Start

Generate all main tables:
```bash
cd /path/to/cje/experiments/arena_10k_simplified/ablations

python -m reporting.cli_generate \
  --results results/all_experiments.jsonl \
  --output tables/ \
  --format latex
```

## Usage Examples

### Generate specific tables
```bash
# Just Table M1 (accuracy by regime)
python -m reporting.cli_generate \
  --results results/all_experiments.jsonl \
  --output tables/ \
  --format latex \
  --tables m1

# Multiple tables
python -m reporting.cli_generate \
  --results results/all_experiments.jsonl \
  --output tables/ \
  --format latex \
  --tables m1,m2,m3
```

### Output formats
```bash
# Markdown for quick viewing
python -m reporting.cli_generate \
  --results results/all_experiments.jsonl \
  --output tables/ \
  --format markdown

# Both LaTeX and Markdown
python -m reporting.cli_generate \
  --results results/all_experiments.jsonl \
  --output tables/ \
  --format both
```

### Filter to specific regimes
```bash
# Small samples + low coverage only
python -m reporting.cli_generate \
  --results results/all_experiments.jsonl \
  --output tables/ \
  --format latex \
  --regimes "250,500;0.05,0.10"

# Multiple regime groups
python -m reporting.cli_generate \
  --results results/all_experiments.jsonl \
  --output tables/ \
  --format latex \
  --regimes "250,500;0.05,0.10|1000,2500,5000;0.25,0.50,1.0"
```

### Additional options
```bash
# Exclude unhelpful policy (not recommended - defaults to including)
python -m reporting.cli_generate \
  --results results/all_experiments.jsonl \
  --output tables/ \
  --format latex \
  --exclude-unhelpful

# Verbose output for debugging
python -m reporting.cli_generate \
  --results results/all_experiments.jsonl \
  --output tables/ \
  --format latex \
  --verbose
```

### Generate additional tables programmatically
```python
from reporting import tables_main, io

# Load the data
df = io.load_results_jsonl('results/all_experiments.jsonl')

# Refuse Rates by Oracle Count (not part of main M1-M3 tables)
refuse_table = tables_main.build_table_refuse_rates(df, by_oracle_count=True)
print("Refuse Rates Table:")
print(refuse_table)
```

## Output Structure

```
tables/
├── main/
│   ├── table_m1_accuracy.tex      # Accuracy & uncertainty by regime
│   ├── table_m2_ess_comparison.tex # ESS comparison (weight calibration effect)
│   └── table_m3_gates.tex         # Gate pass rates & diagnostics
├── figures/
│   └── coverage_vs_width_data.csv # Data for coverage vs width plots
└── summary_statistics.txt         # Key summary statistics
```

## Main Tables

### Table M1: Accuracy & Uncertainty by Regime
- **Purpose**: Show estimator performance across sample size × oracle coverage regimes
- **Columns**: RMSE^d, IS^OA, Coverage %, SE GeoMean, Ranking Metrics, Runtime
- **Key insight**: Performance varies dramatically by regime - no single "winner"

### Table M2: Weight Diagnostics Comparison (SIMCal Effect)
- **Purpose**: Show comprehensive weight diagnostic improvements from SIMCal calibration
- **Compares**: Raw IPS vs Calibrated IPS across all policies
- **Metrics shown**:
  - **ESS (Effective Sample Size)**: Higher is better, measures effective overlap
  - **Weight CV (Coefficient of Variation)**: Lower is better, measures weight stability
  - **Max Weight**: Lower is better, identifies extreme weights
  - **Tail α (Hill Index)**: Higher is better (α>2 needed for CLT), measures tail behavior
- **Key insights**:
  - SIMCal improves ESS by 278% to 21,908%
  - Weight CV reduces by 96-99% (more stable weights)
  - Max weight reduces by 95-99% (fewer extremes)
  - Tail behavior improves from heavy-tailed (α<2) to excellent (α>10)

### Table M3: Gates & Diagnostics
- **Purpose**: Audit trail for reliability gates
- **Columns**: Overlap/Judge/DR/Cap stability pass rates, ESS%, Hill α
- **Key insight**: Which estimators are trustworthy in which regimes

## Additional Tables

### Refuse Rates Table
- **Function**: `build_table_refuse_rates(df, by_oracle_count=True)`
- **Purpose**: Analyze boundary issues by oracle data availability
- **Groups**: Low (12-125), Medium (250-1250), High (2500+) oracle labels
- **Shows**: Out-of-range %, Saturation %, and overall Refuse Rate %
- **Key insight**: Boundary issues decrease with more oracle data

## Architecture

### Modular Pipeline
```
JSONL → Tidy DataFrame → Metrics → Aggregation → Formatting
         (io.py)        (metrics.py) (aggregate.py) (format_latex.py)
```

### Key Components
- **io.py**: Tidy data loading (one row per run × policy)
- **metrics.py**: Pure functions for all metrics (RMSE^d, IS^OA, etc.)
- **aggregate.py**: Groupby operations and paired deltas
- **tables_main.py**: Table builders for M1-M3
- **format_latex.py**: LaTeX formatting with booktabs
- **cli_generate.py**: Unified command-line interface

### Data Model
Each row in the tidy DataFrame represents one (run, policy) pair with:
- Identifiers: run_id, seed, estimator
- Regime: regime_n (sample size), regime_cov (oracle coverage)
- Config: use_calib, rho, outer_cv
- Metrics: est, oracle_truth, se, ci_lo, ci_hi
- Diagnostics: ess_rel, hill_alpha, gates
- Performance: runtime_s

## Key Metrics

### RMSE^d (Debiased RMSE)
```
RMSE^d = sqrt(mean(max(0, (est - oracle)² - var_oracle)))
```
Removes irreducible oracle sampling noise for fair comparison.

### IS^OA (Oracle-Adjusted Interval Score)
```
IS = width + (2/α) × coverage_penalty
```
Balances CI sharpness and calibration in one metric.

### Coverage % (Empirical Coverage)
```
Coverage % = (# CIs covering truth / total CIs) × 100
```
Empirical coverage rate as a percentage:
- **95%**: Perfect calibration (target)
- **< 95%**: Under-coverage (CIs too narrow, risky)
- **> 95%**: Over-coverage (CIs too wide, conservative)
- Best estimator is closest to 95%

### Gate Pass Rates
- **Overlap**: ESS > 10% or Hill α > 2
- **Judge**: Kendall τ > 0.3
- **DR**: Orthogonality CI contains 0
- **Cap Stable**: < 50% weights capped

## Design Principles

1. **No leaderboards**: Regime-based matrices show context-dependent performance
2. **Tidy data**: One source of truth, pure functions over DataFrames
3. **Numerical robustness**: Epsilon guards, IQR-based outlier handling
4. **Statistical rigor**: Bootstrap CIs, Wilcoxon tests, paired comparisons
5. **Journal quality**: Professional LaTeX with booktabs, significance markers

## Migration from v1.0

The old leaderboard-based system has been removed. Key changes:
- `generate_paper_tables.py` → `python -m reporting.cli_generate`
- `paper_tables.py` → split into modular components
- "Leaderboard" → "Accuracy by Regime" (Table M1)
- Aggregate scores removed → focus on regime-specific performance

## Requirements

- Python 3.8+
- pandas, numpy, scipy
- Input: JSONL with experiment results
- Each result must have: estimates, oracle_truths, confidence_intervals

## Troubleshooting

**Empty tables**: Check that results file has successful runs (`"success": true`)

**Missing regimes**: Verify sample_size and oracle_coverage values match your experiments

**Pandas warnings**: Update pandas to latest version or ignore FutureWarnings

**MC diagnostics missing**: Only computed for DR estimators (dr-cpo, stacked-dr, etc.)

## File Structure
```
reporting/
├── __init__.py          # Module exports
├── io.py               # Tidy data loader
├── metrics.py          # Metric computation
├── aggregate.py        # Aggregation utilities
├── tables_main.py      # Main table builders
├── format_latex.py     # LaTeX formatting
├── cli_generate.py     # CLI interface
├── README.md           # This file
└── ../tests/
    └── test_reporting.py # Smoke tests for regression prevention
```

## Recent Improvements (v2.1)

### Correctness Fixes
- **Runtime deduplication**: No longer double-counts runtime across policies
- **DR detection**: Robust regex with word boundaries, handles NaN estimators
- **Variance cap sensitivity**: Uses paired matching (same seed/regime)
- **SE percentage**: Mean of per-pair changes, not ratio of means
- **CI sorting**: Ensures ci_lo < ci_hi for malformed data

### New Features
- **Coverage delta**: Added to Table M2 as "ΔCoverage (pp)"
- **Bootstrap seed**: Optional seed parameter for reproducibility
- **Overall row handling**: Excluded from best/second highlighting
- **Smoke tests**: Comprehensive test suite in tests/test_reporting.py

## Version History

- **v2.1** (current): Production-ready with correctness fixes and tests
- **v2.0**: Modular architecture, regime-based tables, no leaderboards
- **v1.0** (removed): Legacy leaderboard implementation