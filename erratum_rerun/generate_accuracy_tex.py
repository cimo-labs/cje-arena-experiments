#!/usr/bin/env python3
"""Produce the corrected tab:accuracy for the arXiv v2 erratum.

Reads the corrected per-estimator metrics for the six affected rows
(direct, direct+cov, calibrated-ips, calibrated-ips+cov, stacked-dr,
stacked-dr+cov) from the rerun's table_m1 output (reporting.cli_generate on
accuracy_subset.jsonl), substitutes them into the published arxiv
accuracy.tex row set (all other rows keep their v1 values verbatim),
recomputes bold/underline emphasis with the published rules, and writes
tables/accuracy.tex.

The direct rows were rerun with the published analytic inference provenance
pinned (extra.inference_method="cluster_robust" + OUA jackknife), so their
correction is attributable to the OUA x K fix alone, not 0.3.0's bootstrap
default.

Also writes tables/accuracy_rows_corrected.tex containing just the six
corrected rows (unemphasised) for convenience.
"""
import re
from pathlib import Path

BASE = Path(__file__).parent
PUBLISHED_TEX = Path(
    "/Users/eddielandesberg/CIMO/CJE/cje-paper/arxiv_submission/tables/accuracy.tex"
)
M1_RERUN = BASE / "tables_raw" / "main" / "table_m1_accuracy.tex"
OUT_FULL = BASE / "tables" / "accuracy.tex"
OUT_ROWS = BASE / "tables" / "accuracy_rows_corrected.tex"

AFFECTED = [
    "direct",
    "direct+cov",
    "calibrated-ips",
    "calibrated-ips+cov",
    "stacked-dr",
    "stacked-dr+cov",
]

# Column spec: (index, better) where better in {min, max, target95}
# Columns: RMSE^d, IS, Coverage, SE GM, Pairwise, Top-1, tau, Runtime
BETTER = ["min", "min", "target95", "min", "max", "max", "max", "min"]
FMT = ["{:.4f}", "{:.4f}", "{:.1f}", "{:.4f}", "{:.1f}", "{:.1f}", "{:.3f}", "{:.1f}"]


def strip_markup(cell: str) -> str:
    cell = cell.strip()
    cell = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cell)
    cell = re.sub(r"\\underline\{([^}]*)\}", r"\1", cell)
    return cell.strip()


def parse_rows(tex: str):
    """Parse estimator rows from a table tex -> list of (name, [8 floats])."""
    rows = []
    for line in tex.splitlines():
        line = line.strip()
        if not line.endswith(r"\\") or "&" not in line:
            continue
        cells = [c.strip() for c in line[:-2].split("&")]
        name = strip_markup(cells[0])
        if name in ("Estimator", "Regime") or name.startswith("$"):
            continue
        try:
            vals = [float(strip_markup(c)) for c in cells[1:]]
        except ValueError:
            continue
        rows.append((name, vals))
    return rows


def emphasis(rows):
    """Return {(row_idx, col_idx): 'bf'|'ul'} using published rules."""
    marks = {}
    ncols = len(rows[0][1])
    for j in range(ncols):
        col = [(i, r[1][j]) for i, r in enumerate(rows)]
        if BETTER[j] == "min":
            ranked = sorted(col, key=lambda t: t[1])
        elif BETTER[j] == "max":
            ranked = sorted(col, key=lambda t: -t[1])
        else:  # target95
            ranked = sorted(col, key=lambda t: abs(t[1] - 95.0))
        marks[(ranked[0][0], j)] = "bf"
        if len(ranked) > 1:
            marks[(ranked[1][0], j)] = "ul"
    return marks


def fmt_cell(v, j, mark):
    s = FMT[j].format(v)
    if mark == "bf":
        return r"\textbf{" + s + "}"
    if mark == "ul":
        return r"\underline{" + s + "}"
    return s


def main() -> None:
    published = PUBLISHED_TEX.read_text()
    pub_rows = parse_rows(published)
    rerun_rows = dict(parse_rows(M1_RERUN.read_text()))

    missing = [n for n in AFFECTED if n not in rerun_rows]
    if missing:
        raise SystemExit(f"Missing corrected rows in {M1_RERUN}: {missing}")

    merged = []
    for name, vals in pub_rows:
        if name in AFFECTED:
            merged.append((name, rerun_rows[name]))
        else:
            merged.append((name, vals))

    marks = emphasis(merged)

    body_lines = []
    for i, (name, vals) in enumerate(merged):
        cells = [fmt_cell(v, j, marks.get((i, j))) for j, v in enumerate(vals)]
        body_lines.append(name + " & " + " & ".join(cells) + r" \\")

    # Rebuild the published tex, replacing only the data rows between
    # \midrule and \bottomrule.
    pre, rest = published.split("\\midrule", 1)
    _, post = rest.split("\\bottomrule", 1)
    out = pre + "\\midrule\n" + "\n".join(body_lines) + "\n\\bottomrule" + post
    # Update the footnote to note the correction
    out = out.replace(
        "Metrics averaged across all regimes.",
        "Metrics averaged across all regimes. Rows direct($\\pm$cov), "
        "calibrated-ips($\\pm$cov) and stacked-dr($\\pm$cov) rerun with "
        "\\texttt{cje-eval} 0.3.0 (corrected OUA jackknife aggregation, "
        "corrected CalibratedIPS Hajek influence function, fixed two-stage "
        "OUA, real TMLE fluctuation); direct rows keep the published analytic "
        "inference path (cluster-robust SE + OUA jackknife); "
        "all other rows unchanged from v1.",
    )
    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    OUT_FULL.write_text(out)

    with open(OUT_ROWS, "w") as f:
        for name in AFFECTED:
            vals = rerun_rows[name]
            f.write(
                name + " & " + " & ".join(FMT[j].format(v) for j, v in enumerate(vals))
                + " \\\\\n"
            )
    print(f"Wrote {OUT_FULL} and {OUT_ROWS}")


if __name__ == "__main__":
    main()
