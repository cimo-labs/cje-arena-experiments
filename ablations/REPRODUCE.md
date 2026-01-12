# Reproducing Arena Preferences Ablation Results

**Last Validated:** January 2026

This guide documents how to reproduce the CJE vs Rogan-Gladen comparison on Chatbot Arena preferences.

## Quick Validation (< 1 minute)

```bash
cd ablations
python -c "from run_preferences import PreferencesAblation; a = PreferencesAblation(); print('Setup OK')"
```

## Prerequisites

- Python 3.9+
- cje-eval >= 0.2.12
- ~2GB disk space for results

```bash
pip install cje-eval>=0.2.12
python -c "import cje; print(f'cje-eval version: {cje.__version__}')"
```

## Data Integrity Verification

Before running experiments, verify data checksums:

```bash
cd /path/to/cje-arena-experiments
shasum -a 256 data_preferences/preferences_dataset.jsonl data_preferences/responses/*.jsonl
```

**Expected checksums:**

| File | SHA256 |
|------|--------|
| preferences_dataset.jsonl | b4b174964ffc9062...877a8 |
| base_responses.jsonl | 93d36b8440e4fcc5...98445 |
| contrarian_responses.jsonl | 23e28cee99585ca4...018f3f |
| terse_responses.jsonl | 45529345efdcdf8d...55f8b6 |
| verbose_responses.jsonl | 07bcf2d24e6a4804...aeb056 |

## Full Reproduction

### Step 1: Run Continuous CJE Experiments

```bash
cd ablations
python run_preferences.py
```

- **Expected output:** 900 experiments (2 estimators × 3 sample sizes × 5 coverages × 30 seeds)
- **Runtime:** ~45-60 minutes on 8-core CPU
- **Result file:** `results_preferences/all_experiments.jsonl`

### Step 2: Run Binary Baseline (Rogan-Gladen)

```bash
python run_preferences.py --binary
```

- **Expected output:** 450 experiments (3 sample sizes × 5 coverages × 30 seeds)
- **Runtime:** ~2-3 minutes
- **Result file:** `results_preferences/binary_baseline.jsonl`

### Step 3: Generate Report

```bash
python diagnose_preferences.py --report
```

- **Output:** `results_preferences/REPORT.md`

### Step 4: Run Threshold Sweep (Optional)

Tests whether binarization threshold affects results:

```bash
python run_preferences.py --threshold-sweep
```

- **Expected output:** Results for τ ∈ {0.3, 0.4, 0.5, 0.6, 0.7}
- **Runtime:** ~1 minute
- **Result file:** `results_preferences/threshold_sweep.json`
- **Key finding:** Optimal threshold (τ=0.60) achieves RMSE 0.061, still 93% worse than CJE (0.004)

## Expected Results

### File Sizes

```bash
wc -l results_preferences/all_experiments.jsonl    # Should be 900
wc -l results_preferences/binary_baseline.jsonl    # Should be 450
```

### Key Metrics (±tolerance for seed variation)

| Metric | Expected | Tolerance |
|--------|----------|-----------|
| CJE RMSE | 0.004 | ±0.002 |
| Rogan-Gladen RMSE (target policies) | 0.096 | ±0.02 |
| CJE 95% CI Coverage | 96.4% | ±3% |
| Rogan-Gladen 95% CI Coverage | 97.9% | ±3% |
| CJE CI Half-Width | 0.015 | ±0.005 |
| Rogan-Gladen CI Half-Width | 0.192 | ±0.05 |

### Verification Commands

```bash
# Check CJE coverage
python -c "
import json
coverages = []
with open('results_preferences/all_experiments.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r.get('success') and r.get('spec',{}).get('estimator') == 'direct':
            if r.get('ci_covers_oracle') is not None:
                coverages.append(r['ci_covers_oracle'])
print(f'CJE coverage: {sum(coverages)/len(coverages):.1%} (n={len(coverages)})')
"

# Check binary coverage
python -c "
import json
with open('results_preferences/binary_baseline_summary.json') as f:
    s = json.load(f)
print(f'Rogan-Gladen coverage: {s[\"overall_stats\"][\"coverage_95ci\"]:.1%}')
"
```

## Computational Requirements

| Resource | Requirement |
|----------|-------------|
| CPU | 4+ cores recommended (parallelized) |
| Memory | < 4GB RAM |
| Disk | ~500MB for results |
| Total runtime | ~50-60 minutes |

## Troubleshooting

### "ModuleNotFoundError: No module named 'cje'"

```bash
pip install cje-eval>=0.2.12
```

### Results differ significantly from expected

1. Verify data checksums match
2. Check cje-eval version is >= 0.2.12
3. Ensure random seeds are deterministic (check `config_preferences.py`)

### Checkpoint/resume

If interrupted, the runner automatically resumes from checkpoint:
```bash
python run_preferences.py  # Continues from last checkpoint
```

To force fresh run:
```bash
rm results_preferences/checkpoint.jsonl
python run_preferences.py
```

## Configuration Reference

Key parameters in `config_preferences.py`:

```python
EXPERIMENTS_PREFERENCES = {
    "estimators": ["naive-direct", "direct"],
    "sample_sizes": [2000, 5000, 10000],
    "oracle_coverages": [0.01, 0.02, 0.05, 0.10, 0.25],
    "seeds": np.arange(0, 30, 1),
    "inference_method": "bootstrap",
    "n_bootstrap": 500,
    "calibration_policy": "base",
}
```

## Files Reference

| File | Purpose |
|------|---------|
| `config_preferences.py` | Experiment parameters |
| `run_preferences.py` | Main experiment runner |
| `binary_baseline.py` | Lee et al. CI implementation |
| `diagnose_preferences.py` | Report generator |
| `results_preferences/REPORT.md` | Results summary |
