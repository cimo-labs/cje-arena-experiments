"""Aggregate JSONL results into tables/.

Outputs:
  * tables/table1.csv — single-policy: coverage % (with MC-SE), mean width,
    RMSE by arm x m, for each regime x n.
  * tables/multipolicy.csv — per-policy coverage/width/RMSE; pairwise
    sign-error rate among difference-CIs excluding zero; power on the
    vs-unhelpful pairs; NA and transport-audit rates.

Dedup: keeps the LAST occurrence per key (resumable runs may re-append
partially-complete (cell, rep) keys).

Run: .venv/bin/python aggregate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TABLES = HERE / "tables"


def load_jsonl(path: Path, key_cols) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    return df


def table1() -> pd.DataFrame:
    path = RESULTS / "single_policy.jsonl"
    if not path.exists():
        print(f"skip table1: {path} missing")
        return pd.DataFrame()
    df = load_jsonl(path, ["regime", "m", "n", "rep", "estimator"])
    ok = df[df["status"] == "ok"].copy()
    err_rate = (
        df.assign(is_err=df["status"] != "ok")
        .groupby(["regime", "n", "estimator", "m"])["is_err"].mean()
    )

    def _agg(g: pd.DataFrame) -> pd.Series:
        R = len(g)
        cov = g["covered"].mean()
        return pd.Series({
            "R": R,
            "coverage_pct": 100 * cov,
            "mc_se_pct": 100 * np.sqrt(cov * (1 - cov) / R) if R else np.nan,
            "mean_width": g["width"].mean(),
            "rmse": float(np.sqrt(np.mean(np.square(g["err"].astype(float))))),
            "mean_lam_hat": g["lam_hat"].astype(float).mean(),
        })

    t = (
        ok.groupby(["regime", "n", "estimator", "m"])
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    t["error_rate"] = t.set_index(["regime", "n", "estimator", "m"]).index.map(
        err_rate
    ).fillna(0.0)
    t = t.sort_values(["regime", "n", "estimator", "m"]).reset_index(drop=True)
    TABLES.mkdir(exist_ok=True)
    t.to_csv(TABLES / "table1.csv", index=False)
    print(f"wrote {TABLES / 'table1.csv'} ({len(t)} rows)")
    return t


def multipolicy() -> pd.DataFrame:
    path = RESULTS / "multipolicy.jsonl"
    if not path.exists():
        print(f"skip multipolicy: {path} missing")
        return pd.DataFrame()
    df = load_jsonl(path, ["alloc", "B", "rep", "arm", "row_type", "name"])
    out_rows = []

    pol = df[df["row_type"] == "policy"]
    for (alloc, B, arm, name), g in pol.groupby(["alloc", "B", "arm", "name"]):
        ok = g[g["status"] == "ok"]
        na = (g["status"] == "NA_requires_labels").mean()
        R = len(ok)
        cov = ok["covered"].mean() if R else np.nan
        out_rows.append({
            "section": "policy", "alloc": alloc, "B": B, "arm": arm,
            "name": name, "R_ok": R,
            "coverage_pct": 100 * cov if R else np.nan,
            "mc_se_pct": (100 * np.sqrt(cov * (1 - cov) / R)) if R else np.nan,
            "mean_width": ok["width"].mean() if R else np.nan,
            "rmse": (float(np.sqrt(np.mean(np.square(ok["err"].astype(float)))))
                     if R else np.nan),
            "na_rate": na,
        })

    pair = df[df["row_type"] == "pair"]
    for (alloc, B, arm, name), g in pair.groupby(["alloc", "B", "arm", "name"]):
        ok = g[g["status"] == "ok"].copy()
        na = (g["status"] == "NA_requires_labels").mean()
        R = len(ok)
        if R:
            ok["excl0"] = (ok["lo"] > 0) | (ok["hi"] < 0)
            truth = ok["truth"].iloc[0]
            ok["sign_err"] = ok["excl0"] & (
                np.sign(ok["estimate"].astype(float)) != np.sign(truth)
            )
            n_excl = int(ok["excl0"].sum())
            sign_err_rate = (ok.loc[ok["excl0"], "sign_err"].mean()
                             if n_excl else np.nan)
            power = ((ok["excl0"] & ~ok["sign_err"]).mean())
            cov = ok["covered"].mean()
        else:
            n_excl, sign_err_rate, power, cov, truth = 0, np.nan, np.nan, np.nan, np.nan
        out_rows.append({
            "section": "pair", "alloc": alloc, "B": B, "arm": arm,
            "name": name, "R_ok": R, "truth": truth,
            "coverage_pct": 100 * cov if R else np.nan,
            "mean_width": ok["width"].mean() if R else np.nan,
            "n_ci_excl_zero": n_excl,
            "sign_error_rate": sign_err_rate,
            "power_correct_sign": power,
            "na_rate": na,
        })

    tr = df[df["row_type"] == "transport"]
    for (alloc, B, arm, name), g in tr.groupby(["alloc", "B", "arm", "name"]):
        statuses = g["status"].str.replace("audit:", "", regex=False)
        counts = statuses.value_counts(normalize=True)
        out_rows.append({
            "section": "transport", "alloc": alloc, "B": B, "arm": arm,
            "name": name, "R_ok": len(g),
            "pass_rate": counts.get("PASS", 0.0),
            "warn_rate": counts.get("WARN", 0.0),
            "fail_rate": counts.get("FAIL", 0.0),
        })

    t = pd.DataFrame(out_rows)
    TABLES.mkdir(exist_ok=True)
    t.to_csv(TABLES / "multipolicy.csv", index=False)
    print(f"wrote {TABLES / 'multipolicy.csv'} ({len(t)} rows)")
    return t


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    t1 = table1()
    if len(t1):
        print("\n=== table1 (single-policy) ===")
        print(t1.to_string(index=False,
                           float_format=lambda x: f"{x:.4f}"))
    t2 = multipolicy()
    if len(t2):
        print("\n=== multipolicy ===")
        print(t2.to_string(index=False,
                           float_format=lambda x: f"{x:.4f}"))
