#!/usr/bin/env python3
"""Sanity gate + comparison for the UQ rerun.

Loads published and rerun bootstrap_uq.jsonl through the SAME reporting code
(ablations/reporting/tables_bootstrap.py) and prints per-regime coverage for
every arm, plus the sanity-gate check on corrected jackknife coverage
(expected ~92.7-97.3%, mean ~96.0 per erratum-triage.md).

Usage: analyze_uq.py [--rerun PATH] [--csv-dir PATH]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "ablations"))

from reporting.tables_bootstrap import (  # noqa: E402
    load_bootstrap_results,
    generate_coverage_table,
    generate_width_table,
    generate_interval_score_table,
)

PUBLISHED = REPO / "ablations" / "results" / "bootstrap_uq.jsonl"


def coverage_pivot(path: Path) -> pd.DataFrame:
    df = load_bootstrap_results(path)
    return generate_coverage_table(df), df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", type=Path,
                    default=Path(__file__).parent / "results" / "bootstrap_uq.jsonl")
    ap.add_argument("--csv-dir", type=Path,
                    default=Path(__file__).parent / "tables")
    args = ap.parse_args()

    pub_cov, pub_df = coverage_pivot(PUBLISHED)
    new_cov, new_df = coverage_pivot(args.rerun)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.1f}")

    print("=== PUBLISHED coverage (v1, cje ~0.2.10 vintage) ===")
    print(pub_cov.to_string(index=False))
    print("\n=== RERUN coverage (cje-eval 0.3.0) ===")
    print(new_cov.to_string(index=False))

    args.csv_dir.mkdir(parents=True, exist_ok=True)
    pub_cov.to_csv(args.csv_dir / "coverage_published.csv", index=False)
    new_cov.to_csv(args.csv_dir / "coverage_rerun.csv", index=False)

    # Width and interval-score pivots for the rerun (context for DELTAS)
    generate_width_table(new_df).to_csv(args.csv_dir / "width_rerun.csv", index=False)
    generate_interval_score_table(new_df).to_csv(
        args.csv_dir / "interval_score_rerun.csv", index=False)
    generate_width_table(pub_df).to_csv(args.csv_dir / "width_published.csv", index=False)

    # ---- SANITY GATE on the corrected jackknife arm ----
    if "oua_jackknife" not in new_cov.columns:
        print("\nSANITY GATE: SKIP (no oua_jackknife rows yet)")
        return
    jk = new_cov["oua_jackknife"].dropna()
    jk_mean = jk.mean()
    jk_min, jk_max = jk.min(), jk.max()
    print(f"\nSANITY GATE — corrected JK coverage across {len(jk)} regimes:")
    print(f"  min={jk_min:.1f}  max={jk_max:.1f}  mean={jk_mean:.1f}")
    print("  expectation (erratum-triage.md): range ~92.7-100, mean ~96.0")
    ok = (90.0 <= jk_min) and (jk_max <= 100.0) and (93.0 <= jk_mean <= 99.0)
    print(f"  GATE: {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
