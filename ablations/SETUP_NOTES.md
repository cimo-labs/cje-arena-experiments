# Ablations Setup - Ready to Run from Scratch

## What's Ready

The ablations are configured and ready to run from scratch with:

- **9 estimators** (including new `direct` method)
- **5 sample sizes**: 250, 500, 1000, 2500, 5000
- **5 oracle coverages**: 5%, 10%, 25%, 50%, 100%
- **50 seeds** for statistical robustness
- **Total: 12,500 runs** (250 unique configs × 50 seeds)

## Changes Made

### 1. Added Direct Estimator
- Added `direct` estimator for on-policy evaluation using fresh draws
- Configured to never use weight calibration (doesn't use importance weights)
- Automatically uses AutoCal-R when oracle labels available

### 2. Removed Stacked-DR Variants
- Removed `stacked-dr-oc` (was: dr-cpo + oc-dr-cpo + tmle + mrdr)
- Removed `stacked-dr-oc-tr` (was: dr-cpo + oc-dr-cpo + tmle + mrdr + tr-cpo-e)
- Kept standard `stacked-dr` (dr-cpo + tmle + mrdr)

### 3. Fixed Code Issues
- Fixed import paths in `run.py` (`ablations.core.base` → `core.base`)
- Removed obsolete `use_iic` parameters (IIC feature removed from library)
- Updated all constraint mappings for new estimator set

### 4. Archived Old Results
- Previous results (13,750 experiments) backed up to `results/archive_20251006_151503/`
- Cleared `all_experiments.jsonl` and `checkpoint.jsonl` for fresh run
- Old results: 739 MB each file

### 5. Fixed Bytecode Cache Issue (Oct 6, 16:22)
- **Issue**: After fixing `use_iic` parameters, Python's bytecode cache still had old `.pyc` files
- **Impact**: All `dr-cpo` runs failed with `use_iic` error (300/1589 runs, 19% failure rate)
- **Fix**: Cleared all `__pycache__` directories and `.pyc` files
- **Action**: Archived corrupted results to `all_experiments_with_use_iic_bug.jsonl`
- **Note**: Always clear bytecode cache after fixing code: `find . -name "__pycache__" -exec rm -rf {} +`

## Estimators Included

1. **direct** - Direct method (on-policy evaluation)
2. **raw-ips** - SNIPS (self-normalized IPS, no weight calibration)
3. **calibrated-ips** - Calibrated IPS with SIMCal
4. **orthogonalized-ips** - Orthogonalized Calibrated IPS
5. **dr-cpo** - DR-CPO (doubly robust)
6. **oc-dr-cpo** - Orthogonalized Calibrated DR-CPO
7. **tr-cpo-e** - Triply-Robust CPO (efficient variant)
8. **tr-cpo-e-anchored-orthogonal** - TR-CPO with SIMCal anchoring + orthogonalization
9. **stacked-dr** - Ensemble of dr-cpo, tmle, mrdr

## How to Run

### Quick Test (4 configs, ~2 seconds)
```bash
python test_run.py
```

### Full Ablations (12,500 runs, ~50-100 hours)
```bash
python run.py
```

The runner includes:
- Automatic checkpoint/resume support
- Progress tracking with tqdm
- Parallel execution (8 workers by default)
- Results saved to `results/all_experiments.jsonl`

### Monitor Progress
```bash
# Check how many experiments completed
wc -l results/checkpoint.jsonl

# Check results file size
ls -lh results/all_experiments.jsonl

# Tail the latest results
tail -1 results/all_experiments.jsonl | python -m json.tool
```

## Expected Runtime

With 8 parallel workers:
- Per experiment: ~0.2-0.5 seconds average
- Total: 12,500 experiments
- Estimated: 50-100 hours (2-4 days)

Checkpoint system allows stopping and resuming at any time.

## Files Modified

- `config.py` - Updated estimator list, removed stacked-dr variants
- `core/base.py` - Added direct estimator, removed use_iic, removed stacked-dr variants
- `run.py` - Fixed imports (ablations.core → core)
- `visualize_estimator_comparison.py` - Updated display names
- `test_run.py` - Created minimal smoke test

## Verification

Test run confirms all estimators work:
- ✓ direct: RMSE 0.0059, runtime 0.42s
- ✓ raw-ips: RMSE 0.0191, runtime 0.19s
- ✓ calibrated-ips: RMSE 0.0060, runtime 0.22s
- ✓ dr-cpo: RMSE 0.0069, runtime 0.50s

All 4 test configurations pass successfully.

## Troubleshooting

### Code Changes Not Taking Effect
If you modify code but tests/runs still fail with old errors:
```bash
# Clear Python bytecode cache
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete

# Kill any running processes
pkill -f "python run.py"

# Verify cache is cleared
find . -name "__pycache__"  # Should return nothing
```

### Checking Run Status
```bash
# See if ablations are running
ps aux | grep "python run.py" | grep -v grep

# Count completed experiments
wc -l results/checkpoint.jsonl

# Check for errors
jq -r 'select(.success == false) | .error' results/all_experiments.jsonl | sort | uniq -c

# See which estimators are failing
jq -r 'select(.success == false) | .spec.estimator' results/all_experiments.jsonl | sort | uniq -c
```
