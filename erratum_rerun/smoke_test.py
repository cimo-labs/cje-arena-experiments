#!/usr/bin/env python3
"""Smoke test: one run per erratum arm on cje-eval 0.3.0. Nothing is persisted."""
import json
import time

from config import DATA_PATH
from core.base import BaseAblation
from core.schemas import ExperimentSpec


def run(label, estimator, extra, n=250, cov=0.05, seed=0):
    t0 = time.time()
    spec = ExperimentSpec(
        ablation="smoke",
        dataset_path=str(DATA_PATH),
        estimator=estimator,
        sample_size=n,
        oracle_coverage=cov,
        rho=1.0,
        n_seeds=1,
        seed_base=seed,
        extra=extra,
    )
    res = BaseAblation(name="smoke").run_single(spec, seed)
    ok = res.get("success")
    out = {
        "label": label,
        "success": ok,
        "error": res.get("error"),
        "runtime_s": round(time.time() - t0, 1),
    }
    if ok:
        ses = res.get("standard_errors", {})
        out["se_clone"] = ses.get("clone")
        out["ci_clone"] = res.get("confidence_intervals", {}).get("clone")
        out["covers"] = res.get("ci_covers_oracle")
        md = res.get("metadata", {}) or {}
        sec = md.get("se_components", {}) or {}
        out["oracle_var_clone"] = (sec.get("oracle_variance_per_policy") or {}).get("clone")
        out["jk_counts"] = sec.get("oracle_jackknife_counts")
        out["df"] = md.get("degrees_of_freedom")
        out["inference"] = (md.get("inference") or {}).get("method")
        out["reward_calibration_used"] = res.get("reward_calibration_used")
    print(json.dumps(out, default=str))
    return res


if __name__ == "__main__":
    # UQ arms (direct, monotone calibration — extra matches ablate_bootstrap_uq.py)
    run("uq_cluster_robust", "direct",
        {"inference_method": "cluster_robust", "oua_jackknife": False,
         "n_bootstrap": None, "uq_label": "cluster_robust"})
    run("uq_oua_jackknife", "direct",
        {"inference_method": "cluster_robust", "oua_jackknife": True,
         "n_bootstrap": None, "uq_label": "oua_jackknife"})
    run("uq_bootstrap_n200", "direct",
        {"inference_method": "bootstrap", "oua_jackknife": False,
         "n_bootstrap": 200, "uq_label": "bootstrap_n200"})
    # Accuracy rows
    run("calibrated-ips", "calibrated-ips",
        {"use_weight_calibration": True, "reward_calibration_mode": "auto",
         "use_covariates": False, "var_cap": 1.0})
    run("calibrated-ips+cov", "calibrated-ips",
        {"use_weight_calibration": True, "reward_calibration_mode": "auto",
         "use_covariates": True, "var_cap": 1.0})
    run("stacked-dr", "stacked-dr",
        {"use_weight_calibration": True, "reward_calibration_mode": "auto",
         "use_covariates": False, "var_cap": 1.0})
    # Figure config
    run("figure_direct+cov", "direct",
        {"use_weight_calibration": False, "reward_calibration_mode": "auto",
         "use_covariates": True, "var_cap": 1.0,
         "inference_method": "cluster_robust", "oua_jackknife": True},
        n=1000, cov=0.05)
