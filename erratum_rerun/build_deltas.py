#!/usr/bin/env python3
"""Build DELTAS.md — side-by-side published (arXiv v1) vs corrected (cje-eval 0.3.0)
values for every changed cell, with expected-direction checks.

Inputs (all produced earlier in the erratum rerun):
  tables/coverage_published.csv, tables/coverage_rerun.csv   (analyze_uq.py)
  tables/width_published.csv, tables/width_rerun.csv
  tables_raw/main/table_m1_accuracy.tex                      (reporting.cli_generate on rerun subset)
  tables/published_target_rows.csv                           (extracted from published all_experiments.jsonl)
  results/accuracy_subset.jsonl                              (rerun)
  tables/published_figure_rows.csv                           (extracted from published all_experiments.jsonl)
  results/figure_direct_cov.jsonl                            (rerun)
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
OUT = BASE / "DELTAS.md"
POL = ("clone", "parallel_universe_prompt", "premium")

PUB_ACCURACY_ROWS = {
    # From cje-paper/arxiv_submission/tables/accuracy.tex == repo table_m1 (verified identical)
    # RMSE^d, IS, Coverage, SE GM, Pairwise, Top-1, tau, Runtime
    "direct":             [0.0223, 0.0542, 85.0, 0.0068, 91.8, 84.1,  0.837, 0.7],
    "direct+cov":         [0.0256, 0.0568, 86.1, 0.0074, 94.4, 89.5,  0.888, 1.6],
    "calibrated-ips":     [0.0246, 0.5727, 99.1, 0.0963, 47.1, 19.0, -0.059, 0.3],
    "calibrated-ips+cov": [0.0262, 0.6084, 99.2, 0.1027, 46.5, 20.5, -0.070, 0.4],
    "stacked-dr":         [0.0224, 0.0788, 95.6, 0.0125, 91.9, 83.1,  0.838, 8.9],
    "stacked-dr+cov":     [0.0281, 0.0806, 96.1, 0.0128, 92.1, 84.8,  0.842, 38.2],
}
ACC_COLS = ["RMSE^d", "IS", "Coverage %", "SE GM", "Pairwise %", "Top-1 %", "tau", "Runtime (s)"]

PUBLISHED_UQ_TEX = {  # integer cells as printed in v1 uq_comparison.tex
    (250, 0.05): (21, 85, 94, 93), (250, 0.25): (43, 86, 95, 97),
    (500, 0.05): (28, 89, 96, 97), (500, 0.25): (40, 84, 98, 99),
    (1000, 0.05): (21, 75, 95, 95), (1000, 0.25): (35, 83, 95, 95),
    (2500, 0.05): (15, 70, 97, 97), (2500, 0.25): (52, 85, 98, 99),
}


def strip_markup(cell):
    cell = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cell)
    cell = re.sub(r"\\underline\{([^}]*)\}", r"\1", cell)
    return cell.strip()


def parse_m1(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line.endswith(r"\\") or "&" not in line:
            continue
        cells = [strip_markup(c) for c in line[:-2].split("&")]
        try:
            rows[cells[0]] = [float(c) for c in cells[1:]]
        except ValueError:
            continue
    return rows


def load_rerun_target_rows():
    recs = []
    with open(BASE / "results" / "accuracy_subset.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if not d.get("success"):
                continue
            s = d["spec"]; e = s.get("extra") or {}
            for p in POL:
                if p in (d.get("estimates") or {}):
                    recs.append({
                        "estimator": s["estimator"],
                        "use_cov": int(bool(e.get("use_covariates"))),
                        "n": s["sample_size"], "cov": s["oracle_coverage"],
                        "seed": s["seed_base"], "policy": p,
                        "est": d["estimates"][p],
                        "se": (d.get("standard_errors") or {}).get(p),
                    })
    return pd.DataFrame(recs)


def se_ratio_section(md):
    pub = pd.read_csv(BASE / "tables" / "published_target_rows.csv")
    new = load_rerun_target_rows()
    if new.empty:
        md.append("_Rerun accuracy subset not available — SE-ratio check skipped._")
        return
    m = pub.merge(new, on=["estimator", "use_cov", "n", "cov", "seed", "policy"],
                  suffixes=("_pub", "_new"))
    m = m.dropna(subset=["se_pub", "se_new"])
    m["se_ratio"] = m["se_new"] / m["se_pub"]
    m["est_shift"] = (m["est_new"] - m["est_pub"]).abs()
    md.append("### Per-run paired SE ratio (corrected / published), same (estimator, n, coverage, seed, policy)\n")
    md.append("| Row | n pairs | SE ratio median | SE ratio [p10, p90] | max |est shift| |")
    md.append("|---|---|---|---|---|")
    for (est, uc), g in m.groupby(["estimator", "use_cov"]):
        name = est + ("+cov" if uc else "")
        md.append(
            f"| {name} | {len(g)} | {g['se_ratio'].median():.2f}x "
            f"| [{g['se_ratio'].quantile(0.1):.2f}, {g['se_ratio'].quantile(0.9):.2f}] "
            f"| {g['est_shift'].max():.4f} |")
    md.append("")


def load_direct_regime_rows(path, require_analytic=False):
    """Per-regime (use_cov, n, cov) coverage/SE for direct rows of a results jsonl."""
    recs = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if not d.get("success"):
                continue
            s = d["spec"]
            if s.get("estimator") != "direct":
                continue
            e = s.get("extra") or {}
            if require_analytic and e.get("inference_method") not in (None, "cluster_robust"):
                continue
            covers = d.get("ci_covers_oracle") or {}
            ses = d.get("standard_errors") or {}
            for p in POL:
                if p in covers:
                    recs.append({
                        "use_cov": int(bool(e.get("use_covariates"))),
                        "n": s["sample_size"], "cov": s["oracle_coverage"],
                        "covered": bool(covers[p]), "se": ses.get(p),
                    })
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    return (df.groupby(["use_cov", "n", "cov"])
              .agg(coverage_pct=("covered", lambda x: 100.0 * x.mean()),
                   mean_se=("se", "mean"), n_cis=("covered", "size"))
              .reset_index())


def direct_regime_section(md):
    """2b: per-regime JK/OUA coverage for the direct rows, published vs corrected."""
    pub = pd.read_csv(BASE / "tables" / "direct_coverage_published.csv")
    new = load_direct_regime_rows(BASE / "results" / "accuracy_subset.jsonl",
                                  require_analytic=True)
    if new.empty:
        md.append("_Direct rerun rows not available — per-regime coverage skipped._\n")
        return
    md.append("### 2b. Direct rows: per-regime JK/OUA coverage "
              "(3 policies x 50 seeds = 150 CIs per cell, per-seed `ci_covers_oracle`; "
              "slightly different aggregation than the m1 Coverage column)\n")
    md.append("| Regime | direct pub | **direct corrected** | direct+cov pub | **direct+cov corrected** |")
    md.append("|---|---|---|---|---|")
    m = pub.merge(new, on=["use_cov", "n", "cov"], suffixes=("_pub", "_new"))
    piv = {}
    for _, r in m.iterrows():
        piv.setdefault((r["n"], r["cov"]), {})[int(r["use_cov"])] = (
            r["coverage_pct_pub"], r["coverage_pct_new"],
            r["mean_se_pub"], r["mean_se_new"])
    for (n, cov) in sorted(piv):
        c = piv[(n, cov)]
        c0, c1 = c.get(0), c.get(1)
        def fmt(t):
            return (f"{t[0]:.1f} | **{t[1]:.1f}**") if t else "-- | --"
        md.append(f"| n={int(n)}, cov={int(cov*100)}% | {fmt(c0)} | {fmt(c1)} |")
    for uc, name in ((0, "direct"), (1, "direct+cov")):
        mm = m[m["use_cov"] == uc]
        ratio = mm["mean_se_new"] / mm["mean_se_pub"]
        mm_oua = mm[mm["cov"] < 1.0]
        ratio_oua = mm_oua["mean_se_new"] / mm_oua["mean_se_pub"]
        md.append("")
        md.append(
            f"{name}: overall coverage {mm['coverage_pct_pub'].mean():.1f} -> "
            f"**{mm['coverage_pct_new'].mean():.1f}** "
            f"(OUA cells cov<100%: {mm_oua['coverage_pct_pub'].mean():.1f} -> "
            f"{mm_oua['coverage_pct_new'].mean():.1f}); "
            f"mean-SE ratio x{ratio.min():.2f}-x{ratio.max():.2f} "
            f"(median x{ratio.median():.2f}; OUA cells only: median x{ratio_oua.median():.2f}).")
    md.append("")
    md.append("**cov=100% caveat:** `cje-eval` 0.3.0 skips the oracle jackknife at 100% "
              "oracle coverage (no masking -> no OUA term), while the published v1 runs "
              "added a (x K-deflated) OUA term there too. The five cov=100% cells per arm "
              "are therefore CR-only in v2 vs CR + buggy-OUA in v1 and can move slightly "
              "the other way; the OUA-affected cells (cov<100%) isolate the x K fix.\n")


def direct_bootstrap_section(md):
    """2c: bootstrap-inference reference for the direct rows — why it was not rerun."""
    md.append("### 2c. Bootstrap-inference variant (0.3.0 default for direct) — "
              "not rerun on the accuracy grid\n")
    md.append("The task called for the bootstrap variant 'if cheap'. It is not: with the "
              "published rows' `reward_calibration_mode=\"auto\"`, every run in which auto "
              "selects the two-stage calibrator refits it B=500 times — measured "
              "2026-07-07 on this machine: monotone-mode no-cov runs take ~7s (n=250) to "
              "~130s (n=5000), but two-stage no-cov runs take ~820-930s each, and the "
              "direct+cov arm (always two-stage) exceeds ~8 min/run even at n=250. "
              "Full-grid projection: ~7h (no-cov) plus multiple days (+cov) — vs 14 min "
              "for the analytic correction arms. The 37 completed no-cov runs (all "
              "successful, seeds 0-1) are kept in "
              "`results/direct_bootstrap.partial.jsonl`; they are a partial, "
              "non-representative slice and are NOT tabulated.\n")
    md.append("The recorded bootstrap reference for the direct estimator is instead the "
              "UQ-ablation bootstrap arms in Section 1 (same 0.3.0 library, direct "
              "no-cov, B=500/B=2000, 16 regimes n<=2500): rerun coverage 95.3-100.0, "
              "i.e. ~nominal-to-conservative. Caveats: (i) those runs use the UQ "
              "config's default monotone reward calibration rather than auto; (ii) the "
              "0.3.0 direct-mode bootstrap pools oracle labels from fresh draws "
              "(keep-every-oracle-pair, commit ca82e05), so oracle masking does not bind "
              "and bootstrap cells are not apples-to-apples with the published masked "
              "provenance. The correction rows above are the analytic ones.\n")


PROSE_SECTION = r"""## 4. Prose affected by direct-row correction

Every location in `cje-paper/arxiv_submission/` that cites the published direct
coverage numbers (85.0 / 86.1), the direct estimator's analytic-jackknife coverage
band, or statements derived from them. Compiled for the v2 drafting phase; grep
performed 2026-07-07 over all `.tex` sources.

**Note on the printed math:** the delete-one-fold jackknife formula in
`sections/method.tex` (l.132-135) is the CORRECT sum form
`(K-1)/K * sum_k (V^(-k) - Vbar)^2`; the bug was implementation-only (the shipped
library aggregated with a mean instead of a sum, deflating Var_cal by ~x K).
No formula in the paper needs to change — only the empirical numbers and the
claims built on them.

### A. Directly affected: tab:accuracy direct rows (this correction)

1. `tables/accuracy.tex` l.12: `direct & \textbf{0.0223} & \textbf{0.0542} & 85.0 &
   \underline{0.0068} & 91.8 & 84.1 & 0.837 & 0.7 \\` — the IS (0.0542),
   Coverage (85.0) and SE GM (0.0068) cells are x K artifacts; corrected values in
   Section 2 above. Bold/underline emphasis moves accordingly.
2. `tables/accuracy.tex` l.13: `direct+cov & 0.0256 & \underline{0.0568} & 86.1 &
   0.0074 & \textbf{94.4} & \textbf{89.5} & \textbf{0.888} & 1.6 \\` — same three
   cells (0.0568 / 86.1 / 0.0074) affected.

No prose sentence cites 85.0/86.1 directly; the numbers appear only in the table.
The corrected direct rows now cover near-nominally, which REMOVES the paper's
implicit tab:accuracy contrast "direct undercovers analytically (85-86%) while the
recommended bootstrap path achieves ~95%".

### B. Same x K artifact quoted for the direct estimator's analytic jackknife
(tab:uq-comparison / tab:uq-diagnostics; numbers corrected in wave 1 of this
erratum — JK per-regime 70-89 -> 92.7-100.0, overall 82.9 -> 96.0)

3. `sections/experiments.tex` l.90: "\Cref{tab:uq-comparison} compares UQ methods
   for Direct mode: naive cluster-robust SEs achieve only 15--52\% coverage;
   calibration-aware jackknife improves to 70--89\%; bootstrap with
   $\hat\theta_{\mathrm{aug}}$ (AIPW-style residual augmentation) achieves
   ${\sim}$95\% by correcting both variance and bias from calibration
   uncertainty." — the 70--89\% band is the artifact (corrected: ~93--100\%).
4. `sections/experiments.tex` l.226 (Practical takeaways, item 4): "Always use
   bootstrap inference with $\hat\theta_{\mathrm{aug}}$. Standard CIs severely
   under-cover (15--52\%); calibration-aware jackknife improves but still falls
   short (70--89\%); bootstrap with bias correction achieves near-nominal coverage
   (${\sim}$95\%)." — "still falls short" no longer holds; the corrected jackknife
   is ~nominal.
5. `sections/method.tex` l.149: "Combined with bootstrap (refitting the calibrator
   in each replicate), this achieves ${\sim}$95\% coverage across oracle fractions
   (\cref{tab:uq-comparison}), compared to 70--89\% for jackknife-only variance
   estimation." — stale comparison band.
6. `sections/conclusion.tex` l.14: "(ii) \emph{Always use bootstrap inference with
   $\hat\theta_{\mathrm{aug}}$}: naive CIs achieve 0\% coverage; bootstrap with
   bias correction achieves ${\sim}$95\%." — the 0\% and ~95\% figures survive,
   but the "always bootstrap" recommendation was motivated by jackknife
   undercoverage that was an artifact; the v2 rationale needs re-arguing (bias
   correction / small-K df remain valid reasons).
7. `tables/uq_comparison.tex` caption: "Bootstrap with calibrator refit achieves
   near-nominal coverage; cluster-robust and jackknife-only methods undercover."
   + footnote "\textcolor{red}{Red}: $< 90\%$ (undercoverage)." — after
   correction the jackknife column is no longer red/undercovering (cluster-robust
   still is).
8. `tables/uq_diagnostics.tex` (tab:uq-diagnostics): the four "Cal-Aware
   Jackknife" rows (Coverage 80/80/84/87\%; $\bar Z$ +0.48/+0.40/+0.18/-0.05;
   SD$(Z)$ 3.11/2.52/2.09/1.96) are computed with the x K-deflated SEs; coverage
   and the $Z$ moments change under the fix.
9. `sections/appendix_diagnostics.tex` l.81: "\Cref{tab:uq-diagnostics} compares
   UQ methods across oracle fractions, showing that bootstrap with
   $\hat\theta_{\mathrm{aug}}$ achieves ${\sim}$95\% coverage by correcting both
   variance and bias." — comparative framing rests on the stale jackknife rows.

### C. Adjacent claims checked and UNAFFECTED (verified against the rerun)

- `sections/abstract.tex`: "naive confidence intervals on raw judge scores achieve
  0\% coverage (CJE: ${\sim}$95\%)" — naive-direct has no OUA path (x K bug
  can't touch it) and the bootstrap ~95\% reproduces (95.3-100.0 rerun).
- `sections/introduction.tex` l.52, `sections/method.tex` l.82 & l.115,
  `sections/experiments.tex` l.55-56 & l.88, `tables/ablation.tex` (quotes the
  bootstrap 95\% numbers, per its caption), `sections/experiments.tex` l.33
  ("coverage results may be conservative" — oracle-temperature caveat, unrelated
  mechanism): all survive.
- Ranking/accuracy claims tied to the direct rows (91.8/94.4 pairwise, 84.1/89.5
  top-1, tau 0.837/0.888, RMSE^d 0.0223/0.0256, and the "94\% average" in
  `sections/experiments.tex` l.182 / abstract / introduction) are point-estimate
  metrics; per-seed point estimates are unchanged, so these are unaffected.
"""


def fig_decomp(path_csv=None, path_jsonl=None):
    """Return {cov: (base_se, oracle_se, total_se, oua_share)} averaged like the plot script."""
    per_cov = {}
    rows = []
    if path_csv:
        df = pd.read_csv(path_csv)
        for _, r in df.iterrows():
            rows.append((float(r["cov"]), r["seed"], r["policy"],
                         float(r["se_total"]), float(r["oracle_var"])))
    else:
        with open(path_jsonl) as f:
            for line in f:
                d = json.loads(line)
                if not d.get("success"):
                    continue
                s = d["spec"]
                sec = ((d.get("metadata") or {}).get("se_components") or {})
                ovp = sec.get("oracle_variance_per_policy") or {}
                ses = d.get("standard_errors") or {}
                for p in POL:
                    if p in ses and p in ovp:
                        rows.append((float(s["oracle_coverage"]), s["seed_base"], p,
                                     float(ses[p]), float(ovp[p])))
    df = pd.DataFrame(rows, columns=["cov", "seed", "policy", "se_total", "oracle_var"])
    # Replicate plot_oua_decomposition averaging: per run, mean over policies;
    # then mean of variances across runs.
    g = df.groupby(["cov", "seed"]).agg(
        oracle_var=("oracle_var", "mean"), total_se=("se_total", "mean")).reset_index()
    g["total_var"] = g["total_se"] ** 2
    g["base_var"] = np.maximum(0, g["total_var"] - g["oracle_var"])
    for cov, gg in g.groupby("cov"):
        base_se = np.sqrt(gg["base_var"].mean())
        oracle_se = np.sqrt(gg["oracle_var"].mean())
        total_se = gg["total_se"].mean()
        share = gg["oracle_var"].mean() / gg["total_var"].mean() * 100
        per_cov[cov] = (base_se, oracle_se, total_se, share)
    return per_cov


def main():
    md = []
    md.append("# DELTAS — arXiv 2512.11150 v1 (published) vs erratum rerun (cje-eval 0.3.0)\n")
    md.append("Rerun date: 2026-07-06/07. Same data (`data/cje_dataset.jsonl`), same seeds (0-49), "
              "same grid cells as the published runs; the only intended change is the library "
              "(local editable ~v0.2.10-vintage -> PyPI `cje-eval==0.3.0`, which contains the "
              "OUA-jackknife sum-form fix, the two-stage OUA fold-model fix, the real TMLE "
              "fluctuation, and the CalibratedIPS Hajek influence-function fix). Cluster-robust "
              "arms reproduce published values bit-for-bit, so deltas are attributable to the "
              "fixes.\n")

    # ---------------- UQ table ----------------
    md.append("## 1. tab:uq-comparison (coverage %, 3 policies x 50 seeds = 150 CIs per cell)\n")
    pub = pd.read_csv(BASE / "tables" / "coverage_published.csv")
    new = pd.read_csv(BASE / "tables" / "coverage_rerun.csv")
    boot_cols = [c for c in ["bootstrap_n500", "bootstrap_n2000"] if c in new.columns]
    md.append("| Regime | CR pub | CR rerun | JK pub | **JK corrected** | B=500 pub | B=500 rerun | B=2000 pub | B=2000 rerun |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for _, pr in pub.iterrows():
        nr = new[new["regime"] == pr["regime"]]
        nr = nr.iloc[0] if not nr.empty else None
        def gv(row, col):
            return f"{row[col]:.1f}" if row is not None and col in row.index and pd.notna(row[col]) else "--"
        md.append(f"| {pr['regime']} | {pr['cluster_robust']:.1f} | {gv(nr,'cluster_robust')} "
                  f"| {pr['oua_jackknife']:.1f} | **{gv(nr,'oua_jackknife')}** "
                  f"| {pr['bootstrap_n500']:.1f} | {gv(nr,'bootstrap_n500')} "
                  f"| {pr['bootstrap_n2000']:.1f} | {gv(nr,'bootstrap_n2000')} |")
    jk_pub, jk_new = pub["oua_jackknife"], new["oua_jackknife"]
    md.append("")
    md.append(f"Overall (16 regimes): JK published {jk_pub.mean():.1f} -> corrected **{jk_new.mean():.1f}** "
              f"(expected ~96.0); CR published {pub['cluster_robust'].mean():.1f} -> rerun "
              f"{new['cluster_robust'].mean():.1f} (expected: unchanged).")
    ok = jk_new.mean() >= jk_pub.mean() + 5
    md.append(f"**Expected-direction check (JK coverage rises to ~nominal): {'PASS' if ok else 'FAIL'}**\n")
    md.append("**Bootstrap-column caveat:** in `cje-eval` 0.3.0 the direct-mode bootstrap "
              "refit pools oracle labels from fresh draws (keep-every-oracle-pair, commit "
              "ca82e05), so the ablation's oracle masking no longer binds for the bootstrap "
              "arms: their CIs are identical across oracle-coverage levels within each n "
              "(verified per-seed), and per-regime bootstrap cells are not apples-to-apples "
              "with v1. The analytic arms (cluster-robust, cal-aware jackknife) respect the "
              "masking exactly as in v1.\n")
    md.append("Published-table (integer) cells for the 8 shown regimes, v1 -> v2:\n")
    md.append("| Regime | CR | Cal-Aware JK | B=500 | B=2000 |")
    md.append("|---|---|---|---|---|")
    for (n, cov), (cr, jk, b5, b20) in PUBLISHED_UQ_TEX.items():
        regime = f"n={n}, cov={int(cov*100)}%"
        nr = new[new["regime"] == regime]
        if nr.empty:
            continue
        nr = nr.iloc[0]
        def cellv(col, old):
            if col in nr.index and pd.notna(nr[col]):
                v = round(float(nr[col]))
                return f"{old} -> **{v}**" if v != old else f"{old} (=)"
            return f"{old} -> --"
        md.append(f"| {regime} | {cellv('cluster_robust', cr)} | {cellv('oua_jackknife', jk)} "
                  f"| {cellv('bootstrap_n500', b5)} | {cellv('bootstrap_n2000', b20)} |")
    md.append("")

    # CI width context for the JK arm
    wp = pd.read_csv(BASE / "tables" / "width_published.csv")
    wn = pd.read_csv(BASE / "tables" / "width_rerun.csv")
    wm = wp.merge(wn, on="regime", suffixes=("_pub", "_new"))
    if "oua_jackknife_pub" in wm.columns and "oua_jackknife_new" in wm.columns:
        r = (wm["oua_jackknife_new"] / wm["oua_jackknife_pub"])
        md.append(f"Cal-aware JK mean CI width widens x{r.min():.2f}-x{r.max():.2f} "
                  f"(median x{r.median():.2f}) across the 16 regimes — the corrected "
                  f"variance is larger, as expected from the x K fix.\n")

    # ---------------- accuracy table ----------------
    md.append("## 2. tab:accuracy — affected rows only (all other rows unchanged from v1)\n")
    m1 = BASE / "tables_raw" / "main" / "table_m1_accuracy.tex"
    if m1.exists():
        new_rows = parse_m1(m1)
        md.append("| Row | Metric | v1 published | v2 corrected | Δ |")
        md.append("|---|---|---|---|---|")
        for name, pub_vals in PUB_ACCURACY_ROWS.items():
            nv = new_rows.get(name)
            if nv is None:
                md.append(f"| {name} | (all) | -- | MISSING | |")
                continue
            for j, cname in enumerate(ACC_COLS):
                pv, cv = pub_vals[j], nv[j]
                fmt = "{:.4f}" if cname in ("RMSE^d", "IS", "SE GM") else ("{:.3f}" if cname == "tau" else "{:.1f}")
                delta = cv - pv
                md.append(f"| {name} | {cname} | {fmt.format(pv)} | {fmt.format(cv)} | {delta:+.4f} |")
        md.append("")
        # expected-direction: stacked-dr SE GM rises (OUA x K fix);
        # calibrated-ips SE GM FALLS because 0.3.0 also fixes the CalibratedIPS
        # Hajek influence function (commit f99f911: the published IF carried an
        # extra -psi*(w - mean_w) term and was negatively correlated with the
        # true IF; for the poor-overlap tilted policies it overstated SEs x 4-8).
        checks = {}
        for name, pub_vals in PUB_ACCURACY_ROWS.items():
            nv = new_rows.get(name)
            if nv:
                checks[name] = nv[3] / pub_vals[3]
        if checks:
            txt = ", ".join(f"{n}: x{r:.2f}" for n, r in checks.items())
            md.append(f"SE GM ratio (corrected/published): {txt}.\n")
            sdr_up = all(r > 1.0 for n, r in checks.items() if n.startswith("stacked-dr"))
            md.append(f"**Expected-direction check, stacked-dr (OUA x K fix -> SEs rise "
                      f"~1-2.2x): {'PASS' if sdr_up else 'FAIL'}**\n")
            # direct: same mechanism as stacked-dr (OUA x K fix) -> JK/OUA coverage
            # should RISE toward ~95% and SEs widen ~x1.6-2.2. If it moves the other
            # way, the correction is NOT attributable to the x K fix — stop and audit.
            dir_ok = True
            dir_txt = []
            for name in ("direct", "direct+cov"):
                nv = new_rows.get(name)
                if not nv:
                    dir_ok = False
                    continue
                pub_vals = PUB_ACCURACY_ROWS[name]
                cov_up = nv[2] > pub_vals[2] and abs(nv[2] - 95.0) < abs(pub_vals[2] - 95.0)
                se_up = 1.0 < nv[3] / pub_vals[3] < 2.5
                dir_ok = dir_ok and cov_up and se_up
                dir_txt.append(
                    f"{name}: coverage {pub_vals[2]:.1f} -> {nv[2]:.1f}, "
                    f"SE GM x{nv[3] / pub_vals[3]:.2f}")
            md.append(f"**Expected-direction check, direct (OUA x K fix -> JK/OUA coverage "
                      f"rises toward ~95%, SEs widen ~x1.6-2.2): "
                      f"{'PASS' if dir_ok else 'FAIL'}** ({'; '.join(dir_txt)})\n")
            md.append("**Direct point-estimate note:** per-seed point estimates are "
                      "bit-identical to v1 wherever auto-calibration selected the "
                      "monotone mode (direct: 88.6% of paired (run, policy) cells "
                      "have shift exactly 0, p90 1.4e-4, p99 0.010; direct+cov: "
                      "57.3% exactly 0, p90 8.9e-5, p99 0.0005, max 0.0063). The "
                      "remainder are runs where auto selected (or flipped to) the "
                      "two-stage calibrator, whose fold-model fix in 0.3.0 slightly "
                      "perturbs predictions — the same mechanism already seen in the "
                      "stacked-dr rows. The direct max |shift| 0.1297 is a single "
                      "two-stage run (n=1000, cov=5%, seed 34: ~50 oracle labels; "
                      "all 3 policies shift together). The direct-row corrections to "
                      "IS / Coverage / SE GM are driven by the OUA x K fix, not by "
                      "these point shifts.\n")
            md.append("**calibrated-ips deviation from erratum-triage.md's expectation:** "
                      "the triage predicted rising SEs from the OUA fix alone, but "
                      "`cje-eval` 0.3.0 additionally fixes the CalibratedIPS Hajek "
                      "influence function (commit f99f911, 'Fix CalibratedIPS Hajek IF'; "
                      "the shipped IF included a spurious -psi*(w - mean_w) term and was "
                      "negatively correlated (-0.43) with the true IF). Paired per-run "
                      "evidence: clone (perfect overlap) SE ratio median ~1.0 (OUA fix "
                      "only, small), tilted policies (parallel_universe_prompt / premium "
                      "/ unhelpful) median ~0.13-0.25 (IF fix dominates). Point "
                      "estimates are unchanged (max |shift| ~0 for calibrated-ips), so "
                      "the SE drop is purely the inference correction.\n")
        md.append("Note: the Runtime column is environmental (different machine load/library "
                  "internals) and is not an erratum claim.\n")
    else:
        md.append("_Rerun accuracy table not yet generated._\n")
    se_ratio_section(md)
    direct_regime_section(md)
    direct_bootstrap_section(md)

    # ---------------- figure ----------------
    md.append("## 3. OUA-decomposition figure (fig:oua-decomposition, direct+cov, n=1000)\n")
    fp = fig_decomp(path_csv=BASE / "tables" / "published_figure_rows.csv")
    fj = BASE / "results" / "figure_direct_cov.jsonl"
    if fj.exists():
        fn = fig_decomp(path_jsonl=fj)
        md.append("| Oracle % | base SE pub | base SE rerun | oracle SE pub | **oracle SE corrected** | total SE pub | total SE corrected | OUA share pub | **OUA share corrected** |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for cov in sorted(fp):
            b0, o0, t0, s0 = fp[cov]
            if cov in fn:
                b1, o1, t1, s1 = fn[cov]
                md.append(f"| {int(cov*100)}% | {b0:.4f} | {b1:.4f} | {o0:.4f} | **{o1:.4f}** "
                          f"| {t0:.4f} | **{t1:.4f}** | {s0:.0f}% | **{s1:.0f}%** |")
            else:
                md.append(f"| {int(cov*100)}% | {b0:.4f} | -- | {o0:.4f} | -- | {t0:.4f} | -- | {s0:.0f}% | -- (OUA skipped at 100% in 0.3.0) |")
        shares_up = all(fn[c][3] >= fp[c][3] for c in fn if c in fp)
        md.append("")
        md.append(f"**Expected-direction check (OUA share rises; 'calibration uncertainty dominates' "
                  f"strengthens): {'PASS' if shares_up else 'FAIL'}**")
        md.append("Base SE should be ~unchanged (same sampling term); oracle SE should rise ~sqrt(5) "
                  "at K=5. v1's caption claim of '~30% at 100% oracle' relied on cov=1.0 rows that the "
                  "published figure did not plot; cje-eval 0.3.0 intentionally skips the oracle "
                  "jackknife at 100% coverage, so the v2 figure (like v1) shows 5-50%.\n")
    else:
        md.append("_Figure rerun not yet available._\n")

    md.append(PROSE_SECTION)

    OUT.write_text("\n".join(md) + "\n")
    print(f"Wrote {OUT} ({len(md)} lines)")


if __name__ == "__main__":
    main()
