#!/usr/bin/env python3
"""Generate the corrected tab:uq-comparison LaTeX table (published v1 layout).

Layout mirrors cje-paper/arxiv_submission/tables/uq_comparison.tex:
  - rows: 8 regimes (n in {250,500,1000,2500} x oracle {5%,25%})
  - columns: Analytic {Cluster-Robust, Cal-Aware Jackknife}, Bootstrap {B=500, B=2000}
  - integer-rounded coverage, red for <90%
Values come from tables/coverage_rerun.csv (written by analyze_uq.py).
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
CSV = BASE / "tables" / "coverage_rerun.csv"
OUT = BASE / "tables" / "uq_comparison.tex"

SHOWN = [(250, 0.05), (250, 0.25), (500, 0.05), (500, 0.25),
         (1000, 0.05), (1000, 0.25), (2500, 0.05), (2500, 0.25)]
COLS = ["cluster_robust", "oua_jackknife", "bootstrap_n500", "bootstrap_n2000"]


def cell(v: float) -> str:
    s = f"{v:.0f}"
    return r"\textcolor{red}{" + s + "}" if v < 90 else s


def main() -> None:
    df = pd.read_csv(CSV)
    # regime strings like "n=250, cov=5%"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Coverage by UQ Method (target: 95\%), rerun with the corrected "
        r"jackknife variance aggregation (\texttt{cje-eval} 0.3.0). The "
        r"calibration-aware jackknife achieves nominal-to-conservative coverage; "
        r"bootstrap with calibrator refit remains nominal; cluster-robust-only "
        r"SEs undercover.}",
        r"\label{tab:uq-comparison}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Analytic} & \multicolumn{2}{c}{Bootstrap} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"Regime & Cluster-Robust & Cal-Aware Jackknife & $B{=}500$ & $B{=}2000$ \\",
        r"\midrule",
    ]
    missing = []
    for n, cov in SHOWN:
        regime = f"n={n}, cov={int(cov * 100)}%"
        row = df[df["regime"] == regime]
        if row.empty:
            missing.append(regime)
            continue
        row = row.iloc[0]
        cells = []
        for c in COLS:
            if c in row.index and pd.notna(row[c]):
                cells.append(cell(float(row[c])))
            else:
                cells.append("--")
                missing.append(f"{regime}:{c}")
        label = f"$n{{=}}{n}$, {int(cov * 100)}\\% oracle"
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{1mm}",
        r"\footnotesize{\textcolor{red}{Red}: $< 90\%$ (undercoverage). "
        r"Cal-aware jackknife CIs use $t_{K-1}$ critical values ($K$ = calibration "
        r"folds); the analytic jackknife matches bootstrap coverage at "
        r"$\sim$1/2000th the compute. Caveat: in \texttt{cje-eval} 0.3.0 the "
        r"bootstrap refit pools oracle labels from fresh draws (keep-all-pairs), "
        r"so the bootstrap columns no longer vary along the oracle-fraction axis.}",
        r"\end{table}",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")
    if missing:
        print("WARNING — missing cells:", missing)


if __name__ == "__main__":
    main()
