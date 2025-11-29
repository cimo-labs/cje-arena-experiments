# Non-Determinism Sources in CJE Ablations

## Problem Statement
Random variation observed between runs despite fixed data and seed parameter in ablation experiments.

## Root Causes Identified

### 1. **KFold Hardcoded Seeds** (CRITICAL)
**Location**: `cje/calibration/simcal.py:408`, `cje/calibration/judge.py:386`

**Issue**:
```python
# In simcal.py line 408:
kf = KFold(n_splits=self.cfg.n_folds, shuffle=True, random_state=42)

# In judge.py line 386:
kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
```

**Problem**:
- SIMCal **always uses `random_state=42`** regardless of experiment seed
- JudgeCalibrator uses `self.random_seed` BUT this doesn't propagate from ablation seeds
- When ablations run with different seeds (e.g., `seed_base + i`), the KFold splitting remains the same

**Impact**: HIGH
- Cross-validation fold splits are identical across different seed runs
- This defeats the purpose of multi-seed averaging
- Variation comes from other sources, not true resampling

**Fix**:
```python
# Option 1: Thread seed through to SIMCal
class SIMCalConfig:
    random_seed: int = 42  # Add this field

# Option 2: Use the fold_ids parameter (preferred)
# Pass deterministic fold_ids from get_folds_for_prompts() instead
```

### 2. **Oracle Sampling in prepare_dataset()**
**Location**: `ablations/core/base.py:132`

**Issue**:
```python
keep_indices = set(
    random.sample(oracle_indices, min(n_keep, len(oracle_indices)))
)
```

**Status**: ✓ CORRECTLY SEEDED
- Seeds are set at line 98-99: `random.seed(seed)` and `np.random.seed(seed)`
- This IS deterministic given the same seed

### 3. **Dataset Sampling**
**Location**: `ablations/core/base.py:110-115`

**Issue**:
```python
indices = sorted(random.sample(range(len(dataset.samples)), n_samples))
```

**Status**: ✓ CORRECTLY SEEDED
- Uses `random.sample()` after `random.seed(seed)` is called
- Sorting ensures deterministic order

### 4. **NumPy RandomState in get_folds_with_oracle_balance()**
**Location**: `cje/data/folds.py:124-126`

**Issue**:
```python
rng = np.random.RandomState(seed)
oracle_indices = oracle_indices.copy()
rng.shuffle(oracle_indices)
```

**Status**: ✓ CORRECTLY SEEDED
- Uses explicit RandomState with seed parameter
- This IS deterministic

### 5. **Dictionary Iteration Order**
**Status**: ✓ NOT AN ISSUE
- Python 3.7+ guarantees dict insertion order
- All dict iterations are deterministic

### 6. **Hash-based Fold Assignment**
**Location**: `cje/data/folds.py:45-47`

**Issue**:
```python
hash_input = f"{prompt_id}-{seed}-{n_folds}".encode()
hash_bytes = hashlib.blake2b(hash_input, digest_size=8).digest()
return int.from_bytes(hash_bytes, "big") % n_folds
```

**Status**: ✓ DETERMINISTIC
- hashlib produces deterministic hashes
- This is correctly designed

### 7. **Outer CV Seeds in Estimators**
**Location**: `ablations/core/base.py:201, 210, 220`

**Issue**:
```python
CalibratedIPS(
    use_outer_cv=True,
    n_outer_folds=5,
    outer_cv_seed=42,  # ← HARDCODED
)
```

**Problem**: **HARDCODED SEED = 42**
- Every ablation run uses outer_cv_seed=42
- OUA jackknife clustering is identical across seeds
- This reduces true variation between seed runs

**Impact**: MEDIUM-HIGH
- Affects OUA uncertainty quantification
- Particularly impacts partial oracle coverage scenarios

**Fix**:
```python
# Pass the experiment seed through
outer_cv_seed=seed,  # Use the ablation seed
```

## Summary of Issues

| Issue | Location | Severity | Correctly Seeded? |
|-------|----------|----------|-------------------|
| KFold in SIMCal | simcal.py:408 | **CRITICAL** | ❌ Always 42 |
| KFold in JudgeCalibrator | judge.py:386 | HIGH | ⚠️ Needs propagation |
| Outer CV seed | base.py:201+ | MEDIUM-HIGH | ❌ Always 42 |
| Oracle sampling | base.py:132 | - | ✓ Yes |
| Dataset sampling | base.py:110 | - | ✓ Yes |
| Fold assignment | folds.py:45 | - | ✓ Yes |

## Recommended Fixes (Priority Order)

### Fix 1: Replace KFold with Deterministic Fold Assignment (RECOMMENDED)

**Instead of letting SIMCal/JudgeCalibrator create folds internally:**

```python
# In ablation code (base.py), compute folds ONCE:
from cje.data.folds import get_folds_for_dataset

# Before calling calibrate_dataset or estimator.fit:
fold_ids = get_folds_for_dataset(dataset, n_folds=5, seed=seed)

# Pass fold_ids explicitly:
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    fold_ids=fold_ids,  # ← Use deterministic folds
    ...
)
```

**Benefits**:
- Folds are stable across components (SIMCal, JudgeCalibrator, outcome models)
- Folds use the **experiment seed**, not hardcoded 42
- True variation between seed runs

### Fix 2: Thread Seed Through to Estimators

```python
# In base.py create_estimators(), pass seed:
"calibrated-ips": lambda s: CalibratedIPS(
    use_outer_cv=True,
    n_outer_folds=5,
    outer_cv_seed=s,  # ← Use experiment seed, not 42
    var_cap=var_cap,
),
```

**Change needed**: The lambda receives `seed` parameter.

### Fix 3: Verify JudgeCalibrator Seed Propagation

Check that when `calibrate_dataset()` is called, the `random_seed` parameter reaches JudgeCalibrator:

```python
# In ablation code:
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    random_seed=seed,  # ← Does this reach JudgeCalibrator.__init__?
    ...
)
```

**Verify**: Trace through `calibrate_dataset()` → `JudgeCalibrator.__init__()` to ensure seed propagates.

## Testing Recommendations

### Test 1: Verify Fold Determinism
```python
# Run same experiment with same seed twice
seed = 42
result1 = run_experiment(seed=seed)
result2 = run_experiment(seed=seed)

assert np.allclose(result1['estimates'], result2['estimates'])
```

### Test 2: Verify Seed Variation
```python
# Run with different seeds, should get DIFFERENT folds
result_seed_0 = run_experiment(seed=0)
result_seed_1 = run_experiment(seed=1)

# Estimates should differ due to different CV folds
assert not np.allclose(result_seed_0['estimates'], result_seed_1['estimates'])
```

### Test 3: Check Fold Assignments
```python
# Verify KFold uses correct seed
from cje.calibration.simcal import SIMCal
from cje.data.folds import get_folds_for_dataset

# Check if passing fold_ids prevents KFold random generation
fold_ids = get_folds_for_dataset(dataset, seed=999)
# SIMCal should use these, not generate its own
```

## Implementation Priority

1. **HIGH PRIORITY**: Fix KFold seeds in SIMCal and JudgeCalibrator
   - Either pass fold_ids explicitly OR thread random_seed through

2. **MEDIUM PRIORITY**: Fix outer_cv_seed in estimator creation
   - Change from 42 to experiment seed

3. **LOW PRIORITY**: Add determinism tests to CI
   - Ensure future changes don't break reproducibility

## Expected Impact

After fixes:
- ✅ **Same seed** → **Identical results** (bit-for-bit reproducibility)
- ✅ **Different seeds** → **Different CV folds** → **True variation**
- ✅ **Multi-seed averaging** → **Properly represents uncertainty**

Current behavior:
- ⚠️ **Same seed** → Different results (non-deterministic)
- ❌ **Different seeds** → Similar results (same CV folds, not true resampling)

## Additional Notes

### PYTHONHASHSEED
Not currently set. Could add for extra safety:
```bash
export PYTHONHASHSEED=0
```
But Python 3.7+ dict order is already stable, so this is optional.

### Parallel Execution
Check if parallel=True introduces non-determinism:
```python
# In run.py:
parallel=not args.sequential
```

Thread-safety note: Each worker gets its own seed, so this should be fine IF seeds are different per worker.

## References

- Fold management: `cje/data/folds.py`
- SIMCal: `cje/calibration/simcal.py`
- JudgeCalibrator: `cje/calibration/judge.py`
- Ablation runner: `ablations/core/base.py`
