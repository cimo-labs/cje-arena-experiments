"""Figures from tables/results.

  * fig1_<regime>_n<n>.png — two panels, log-x m:
      left: coverage vs m (95% nominal line, MC error bars),
      right: mean width vs m. CJE and cross-PPI visually paired.
  * calibration_curve.png — full-corpus isotonic judge->oracle curve
    (base policy + all-policies pooled).
  * lam_hat_hist.png — PPI++ lam_hat distribution at m in {10, 25}.

Run: .venv/bin/python figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

import data_prep

HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
RESULTS = HERE / "results"
FIGS = HERE / "figs"

# CJE and cross-PPI share a color (visually paired), different linestyle.
STYLE = {
    "cje_calibrated":       dict(color="#c0392b", ls="-",  marker="o", label="CJE (calibrated_mean_ci)"),
    "cross_ppi_cje_folds":  dict(color="#c0392b", ls="--", marker="s", label="cross-PPI (CJE isotonic folds)"),
    "ppi_pp":               dict(color="#2980b9", ls="-",  marker="^", label="PPI++ (lam=None)"),
    "ppi_pp_t":             dict(color="#2980b9", ls="--", marker="v", label="PPI++ (t-adjusted)"),
    "ppi_vanilla":          dict(color="#8e44ad", ls="-",  marker="d", label="PPI (lam=1)"),
    "ppi_pp_weighted":      dict(color="#16a085", ls="-",  marker="P", label="PPI++ weighted"),
    "classical":            dict(color="#7f8c8d", ls="-",  marker="x", label="classical (labels only)"),
    "postpi_ols_ones":      dict(color="#d35400", ls=":",  marker="*", label="PostPI (X=ones)"),
    "naive_judge_t":        dict(color="#2c3e50", ls=":",  marker=".", label="naive judge t-mean"),
}


def fig1():
    t = pd.read_csv(TABLES / "table1.csv")
    FIGS.mkdir(exist_ok=True)
    for (regime, n), g in t.groupby(["regime", "n"]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for est, ge in g.groupby("estimator"):
            ge = ge.sort_values("m")
            st = STYLE.get(est, dict(label=est))
            ax1.errorbar(ge["m"], ge["coverage_pct"], yerr=ge["mc_se_pct"],
                         capsize=3, **st)
            ax2.plot(ge["m"], ge["mean_width"], **st)
        ax1.axhline(95, color="k", lw=1, ls="-", alpha=0.5)
        ax1.set_xscale("log")
        ax1.set_xticks(sorted(g["m"].unique()))
        ax1.get_xaxis().set_major_formatter(ScalarFormatter())
        ax1.set_xlabel("m (oracle labels)")
        ax1.set_ylabel("coverage (%)")
        ax1.set_title(f"coverage vs m — {regime}, n={n}")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xticks(sorted(g["m"].unique()))
        ax2.get_xaxis().set_major_formatter(ScalarFormatter())
        ax2.set_xlabel("m (oracle labels)")
        ax2.set_ylabel("mean CI width")
        ax2.set_title(f"width vs m — {regime}, n={n}")
        ax1.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        out = FIGS / f"fig1_{regime}_n{n}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


def calibration_curve():
    from sklearn.isotonic import IsotonicRegression

    FIGS.mkdir(exist_ok=True)
    corpus = data_prep.load_corpus()
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, df in (
        ("base policy", corpus[corpus["policy"] == "base"]),
        ("all policies pooled", corpus),
    ):
        S = df["judge_score"].to_numpy()
        Y = df["oracle_label"].to_numpy()
        iso = IsotonicRegression(out_of_bounds="clip").fit(S, Y)
        grid = np.linspace(0, 1, 401)
        ax.plot(grid, iso.predict(grid), label=f"isotonic fit — {label}")
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="identity")
    ax.set_xlabel("judge score S")
    ax.set_ylabel("oracle label Y (isotonic E[Y|S])")
    ax.set_title("Full-corpus judge->oracle calibration curve")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIGS / "calibration_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def lam_hat_hist():
    path = RESULTS / "single_policy.jsonl"
    if not path.exists():
        print("skip lam_hat_hist: no single_policy.jsonl")
        return
    rows = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (r["estimator"] == "ppi_pp" and r["status"] == "ok"
                    and r.get("lam_hat") is not None and r["m"] in (10, 25)):
                rows.append(r)
    if not rows:
        print("skip lam_hat_hist: no ppi_pp rows at m=10/25")
        return
    df = pd.DataFrame(rows)
    FIGS.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, m in zip(axes, (10, 25)):
        for (regime, n), g in df[df["m"] == m].groupby(["regime", "n"]):
            ax.hist(g["lam_hat"].astype(float), bins=30, alpha=0.5,
                    label=f"{regime}, n={n}")
        ax.set_title(f"lam_hat at m={m}")
        ax.set_xlabel("lam_hat")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("count")
    fig.tight_layout()
    out = FIGS / "lam_hat_hist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    if (TABLES / "table1.csv").exists():
        fig1()
    else:
        print("skip fig1: tables/table1.csv missing (run aggregate.py)")
    calibration_curve()
    lam_hat_hist()
