# Quick Start

## Documentation Guide

| Goal | Document |
|------|----------|
| Understand research | [ANALYST_HANDOFF.md](ANALYST_HANDOFF.md) |
| Reproduce results | [REPRODUCE.md](REPRODUCE.md) |
| Modify experiments | [README.md](README.md) |
| Generate tables | [reporting/README.md](reporting/README.md) |

## Key Commands

| Task | Command | Time |
|------|---------|------|
| Full CJE experiment | `python run_preferences.py` | ~45min |
| Binary baseline | `python run_preferences.py --binary` | ~3min |
| Threshold sweep | `python run_preferences.py --threshold-sweep` | ~1min |
| Generate report | `python diagnose_preferences.py --report` | ~10s |

## Key Results (Arena Preferences)

| Metric | CJE | Rogan-Gladen | Winner |
|--------|-----|--------------|--------|
| RMSE | 0.004 | 0.096 | CJE (95% better) |
| 95% CI Coverage | 96.4% | 97.9% | Tie |
| CI Half-Width | 0.015 | 0.192 | CJE (13x narrower) |

## File Structure

```
ablations/
├── run_preferences.py      # Main experiment runner
├── binary_baseline.py      # Lee et al. CI implementation
├── diagnose_preferences.py # Report generator
├── config_preferences.py   # Experiment parameters
├── results_preferences/    # Output files
└── _archive/              # Exploratory scripts
```
