#!/usr/bin/env python3
"""
Erratum rerun driver (arXiv 2512.11150 v2) — affected accuracy-table rows + OUA figure.

Reruns, on cje-eval==0.3.0 (fixed OUA jackknife sum-form, fixed two-stage OUA,
real TMLE fluctuation), EXACTLY the published grid cells that feed:

  * tab:accuracy rows calibrated-ips(+cov) and stacked-dr(+cov)
      - estimator in {calibrated-ips, stacked-dr}
      - use_weight_calibration=True (REQUIRES_CALIBRATION, as in run.py)
      - use_covariates in {False, True}
      - sample_size in {250, 500, 1000, 2500, 5000}
      - oracle_coverage in {0.05, 0.10, 0.25, 0.50, 1.00}
      - seeds 0..49, reward_calibration_mode="auto", var_cap=1.0
    -> 5000 runs total, matching all_experiments.jsonl's 4x1250 target rows.

  * tab:accuracy rows direct(+cov)  [erratum extension, 2026-07]
      - estimator=direct, use_weight_calibration=False (NEVER_CALIBRATED)
      - same grid/seeds as above -> 2500 runs
      - extra.inference_method="cluster_robust", extra.oua_jackknife=True
        NOTE: like the figure rows, the published direct accuracy rows carry
        no `inference` metadata block and have se_components.oracle_jackknife_
        counts (verified in all_experiments.jsonl) — i.e. analytic cluster-
        robust SE + OUA jackknife. cje-eval 0.3.0 defaults Direct to
        inference_method="bootstrap" (which also pools oracle labels across
        coverage levels), so we pin cluster_robust to keep the correction
        attributable to the OUA x K fix alone.
      Run with: --job accuracy --estimators direct

  * OPTIONAL reference arm: --job direct-bootstrap
      - same grid/seeds, estimator=direct, no-cov arm only by default
        (--bootstrap-arms; the +cov bootstrap refits the two-stage calibrator
        B times per run, ~8+ min/run — prohibitive on the full grid)
      - extra.inference_method="bootstrap", oua_jackknife=False,
        n_bootstrap=500 (as the UQ ablation's bootstrap_n500 arm)
      - written to results/direct_bootstrap.jsonl (kept OUT of
        accuracy_subset.jsonl: reporting dedup keys would collide with the
        analytic direct rows)

  * the OUA-decomposition appendix figure (oua_vs_base_variance.png)
      - estimator=direct, use_covariates=True, n=1000, all 5 coverages, seeds 0..49
      - extra.inference_method="cluster_robust", extra.oua_jackknife=True
        NOTE: the published figure's rows were produced by a library vintage in
        which CalibratedDirectEstimator had no bootstrap inference: the effective
        path was analytic cluster-robust SE + OUA jackknife (verified from the
        published rows' metadata: no `inference` block, oracle_jackknife_counts
        K=5). Current core/base.py defaults Direct to inference_method="bootstrap",
        so we pin cluster_robust explicitly to reproduce the published provenance
        with only the library bug fixed.
    -> 250 runs.

Usage:
    python run_erratum_accuracy.py --job accuracy [--estimators calibrated-ips,stacked-dr] [--workers 8]
    python run_erratum_accuracy.py --job figure   [--workers 8]
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from config import DATA_PATH
from core.base import BaseAblation
from core.schemas import ExperimentSpec

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent

# Published grid (ablations/config.py EXPERIMENTS) restricted to erratum scope
SAMPLE_SIZES = [250, 500, 1000, 2500, 5000]
ORACLE_COVERAGES = [0.05, 0.10, 0.25, 0.50, 1.00]
SEEDS = list(range(50))
VAR_CAP = 1.0

ACCURACY_ESTIMATORS = ["calibrated-ips", "stacked-dr"]  # both REQUIRES_CALIBRATION
NEVER_CALIBRATED = {"direct"}  # direct never uses weight calibration (config.py)


def accuracy_specs(estimators: List[str]) -> List[Tuple[ExperimentSpec, int]]:
    specs = []
    for seed in SEEDS:  # seeds outermost, as in run.py
        for estimator in estimators:
            for sample_size in SAMPLE_SIZES:
                for oracle_coverage in ORACLE_COVERAGES:
                    for use_covariates in [False, True]:
                        extra = {
                            "use_weight_calibration": estimator
                            not in NEVER_CALIBRATED,
                            "reward_calibration_mode": "auto",
                            "use_covariates": use_covariates,
                            "var_cap": VAR_CAP,
                        }
                        if estimator == "direct":
                            # Reproduce the published analytic provenance
                            # (cluster-robust SE + OUA jackknife); see module
                            # docstring. 0.3.0 would otherwise default direct
                            # to bootstrap inference.
                            extra["inference_method"] = "cluster_robust"
                            extra["oua_jackknife"] = True
                        spec = ExperimentSpec(
                            ablation="unified",
                            dataset_path=str(DATA_PATH),
                            estimator=estimator,
                            sample_size=sample_size,
                            oracle_coverage=oracle_coverage,
                            rho=VAR_CAP,
                            n_seeds=1,
                            seed_base=seed,
                            extra=extra,
                        )
                        specs.append((spec, seed))
    return specs


def direct_bootstrap_specs(
    n_bootstrap: int, cov_arms: Tuple[bool, ...] = (False,)
) -> List[Tuple[ExperimentSpec, int]]:
    """Reference arm: direct with 0.3.0 bootstrap inference (B=n_bootstrap).

    NOT a correction row — 0.3.0's direct-mode bootstrap pools oracle labels
    from fresh draws (keep-every-oracle-pair), so oracle masking does not bind
    and per-coverage cells are not apples-to-apples with v1 (see DELTAS.md).

    Default cov_arms=(False,): the +cov arm's bootstrap refits the two-stage
    calibrator B times per run (~8+ min per run even at n=250, measured
    2026-07-07) and is computationally prohibitive on the full grid, so only
    the no-cov reference is recorded.
    """
    specs = []
    for seed in SEEDS:
        for sample_size in SAMPLE_SIZES:
            for oracle_coverage in ORACLE_COVERAGES:
                for use_covariates in cov_arms:
                    spec = ExperimentSpec(
                        ablation="unified",
                        dataset_path=str(DATA_PATH),
                        estimator="direct",
                        sample_size=sample_size,
                        oracle_coverage=oracle_coverage,
                        rho=VAR_CAP,
                        n_seeds=1,
                        seed_base=seed,
                        extra={
                            "use_weight_calibration": False,
                            "reward_calibration_mode": "auto",
                            "use_covariates": use_covariates,
                            "var_cap": VAR_CAP,
                            "inference_method": "bootstrap",
                            "oua_jackknife": False,
                            "n_bootstrap": n_bootstrap,
                        },
                    )
                    specs.append((spec, seed))
    return specs


def figure_specs() -> List[Tuple[ExperimentSpec, int]]:
    specs = []
    for seed in SEEDS:
        for oracle_coverage in ORACLE_COVERAGES:
            spec = ExperimentSpec(
                ablation="unified",
                dataset_path=str(DATA_PATH),
                estimator="direct",
                sample_size=1000,
                oracle_coverage=oracle_coverage,
                rho=VAR_CAP,
                n_seeds=1,
                seed_base=seed,
                extra={
                    "use_weight_calibration": False,
                    "reward_calibration_mode": "auto",
                    "use_covariates": True,
                    "var_cap": VAR_CAP,
                    # Reproduce the published analytic + OUA-jackknife provenance
                    "inference_method": "cluster_robust",
                    "oua_jackknife": True,
                },
            )
            specs.append((spec, seed))
    return specs


def exp_id(spec: Dict[str, Any]) -> str:
    extra = spec.get("extra", {})
    return "_".join(
        [
            str(spec.get("estimator", "")),
            str(spec.get("sample_size", "")),
            str(spec.get("oracle_coverage", "")),
            str(extra.get("use_weight_calibration", "")),
            str(extra.get("use_covariates", "")),
            str(extra.get("inference_method", "")),
            str(spec.get("seed_base", "")),
        ]
    )


def load_checkpoint(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("success"):
                    done.add(exp_id(r.get("spec", {})))
        logger.info(f"Loaded {len(done)} completed experiments from checkpoint")
    return done


def worker(spec_dict: Dict[str, Any], seed: int) -> Dict[str, Any]:
    ablation = BaseAblation(name="erratum_rerun")
    spec = ExperimentSpec(**spec_dict)
    try:
        return ablation.run_single(spec, seed)
    except Exception as e:  # pragma: no cover
        return {"spec": spec.to_dict(), "success": False, "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Erratum accuracy/figure reruns")
    parser.add_argument(
        "--job", choices=["accuracy", "figure", "direct-bootstrap"], required=True
    )
    parser.add_argument(
        "--estimators",
        type=str,
        default=",".join(ACCURACY_ESTIMATORS),
        help="Comma-separated estimators for --job accuracy",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=500,
        help="Bootstrap replicates for --job direct-bootstrap",
    )
    parser.add_argument(
        "--bootstrap-arms",
        choices=["nocov", "cov", "both"],
        default="nocov",
        help="Covariate arms for --job direct-bootstrap (cov is ~8+ min/run)",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.job == "accuracy":
        estimators = [e.strip() for e in args.estimators.split(",")]
        specs = accuracy_specs(estimators)
        results_path = BASE_DIR / "results" / "accuracy_subset.jsonl"
    elif args.job == "direct-bootstrap":
        cov_arms = {"nocov": (False,), "cov": (True,), "both": (False, True)}[
            args.bootstrap_arms
        ]
        specs = direct_bootstrap_specs(args.n_bootstrap, cov_arms)
        results_path = BASE_DIR / "results" / "direct_bootstrap.jsonl"
    else:
        specs = figure_specs()
        results_path = BASE_DIR / "results" / "figure_direct_cov.jsonl"

    checkpoint_path = results_path.with_suffix(".checkpoint.jsonl")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_checkpoint(checkpoint_path)
    todo = [(s, seed) for s, seed in specs if exp_id(s.to_dict()) not in done]

    logger.info("=" * 60)
    logger.info(f"ERRATUM RERUN — job={args.job}")
    logger.info(f"Total specs: {len(specs)} | remaining: {len(todo)} | done: {len(done)}")
    logger.info(f"Results -> {results_path}")
    logger.info("=" * 60)

    conv = BaseAblation(name="erratum_rerun")  # for _convert_numpy
    n_ok = 0
    n_fail = 0
    t0 = time.time()

    def save(result: Dict[str, Any]) -> None:
        result = conv._convert_numpy(result)
        with open(checkpoint_path, "a") as f:
            f.write(json.dumps(result) + "\n")
        with open(results_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    with tqdm(total=len(todo), desc=f"{args.job} runs") as pbar:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(worker, spec.to_dict(), seed): (spec, seed)
                for spec, seed in todo
            }
            for fut in as_completed(futures):
                spec, seed = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:  # pragma: no cover
                    result = {"spec": spec.to_dict(), "success": False, "error": str(e)}
                save(result)
                if result.get("success"):
                    n_ok += 1
                else:
                    n_fail += 1
                    logger.error(f"FAILED: {result.get('error')}")
                pbar.set_postfix(ok=n_ok, fail=n_fail)
                pbar.update(1)

    logger.info(
        f"DONE job={args.job}: ok={n_ok} fail={n_fail} "
        f"wall={time.time() - t0:.0f}s -> {results_path}"
    )


if __name__ == "__main__":
    main()
