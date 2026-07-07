"""Uniform estimator adapters for the single-policy comparison.

Every arm is `fn(S_all, Y_all, labeled_idx, alpha, seed, weights=None) ->
dict(est, lo, hi, extras)` where
  * S_all: (n,) judge scores of the drawn sample,
  * Y_all: (n,) oracle labels of the drawn sample (arms may only touch
    Y_all[labeled_idx] — enforced by construction, never by imputation),
  * labeled_idx: positions of the oracle-labeled rows,
  * weights: None (uniform regime) or dict(pi=(n,) exact inclusion probs)
    (stratified regime; only the weighted arm uses it).

alpha == 0.05 is ASSERTED at every adapter entry (ppi_py defaults alpha=0.1 —
a silent-mismatch trap; see validate.py check b).

PPI intervals are NOT clipped to [0, 1] (reported as shipped).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy import stats

from ppi_py import (
    classical_mean_ci,
    crossppi_mean_ci,
    crossppi_mean_pointestimate,
    ppi_mean_ci,
    ppi_mean_pointestimate,
    postprediction_ols_ci,
)
from ppi_py.ppi import _calc_lam_glm  # pinned wheel (ppi-python==0.2.3)
from ppi_py.utils import construct_weight_vector, reshape_to_2d

from cje import calibrated_mean_ci
from cje.data.folds import get_folds_for_prompts

logging.getLogger("cje").setLevel(logging.ERROR)

ALPHA = 0.05


def _check_alpha(alpha: float) -> None:
    assert alpha == ALPHA, (
        f"alpha={alpha} != pre-registered {ALPHA}. ppi_py defaults to "
        f"alpha=0.1 — every call must pass alpha explicitly."
    )


def _f(x) -> float:
    return float(np.asarray(x).squeeze())


def _split(S_all: np.ndarray, Y_all: np.ndarray, labeled_idx: np.ndarray):
    """Disjoint labeled/unlabeled split of the drawn sample."""
    n = len(S_all)
    mask = np.zeros(n, dtype=bool)
    mask[labeled_idx] = True
    Y_lab = Y_all[mask]
    S_lab = S_all[mask]
    S_unl = S_all[~mask]
    return Y_lab, S_lab, S_unl, mask


def lam_hat_ppi(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
    w: Optional[np.ndarray] = None,
    w_unlabeled: Optional[np.ndarray] = None,
) -> float:
    """Reproduce ppi_mean_ci's internal lam=None estimate exactly (for
    recording lam_hat; the interval itself always comes from ppi_mean_ci)."""
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    Y2 = reshape_to_2d(Y)
    Yhat2 = reshape_to_2d(Yhat)
    Yhat_unl2 = reshape_to_2d(Yhat_unlabeled)
    d = Y2.shape[1]
    w2 = construct_weight_vector(n, w, vectorized=True)
    w_unl2 = construct_weight_vector(N, w_unlabeled, vectorized=True)
    pointest = ppi_mean_pointestimate(
        Y2, Yhat2, Yhat_unl2, lam=1, w=w2, w_unlabeled=w_unl2
    )
    grads = w2 * (Y2 - pointest)
    grads_hat = w2 * (Yhat2 - pointest)
    grads_hat_unlabeled = w_unl2 * (Yhat_unl2 - pointest)
    lam = _calc_lam_glm(
        grads,
        grads_hat,
        grads_hat_unlabeled,
        np.eye(d),
        coord=None,
        clip=True,
        optim_mode="overall",
    )
    return _f(lam)


# ---------------------------------------------------------------------------
# Arms 1-6, 9 (non-CJE)
# ---------------------------------------------------------------------------


def arm_naive_judge(S_all, Y_all, labeled_idx, alpha, seed, weights=None):
    """(1) t-interval on the mean of ALL n raw judge scores (no oracle use)."""
    _check_alpha(alpha)
    n = len(S_all)
    est = _f(np.mean(S_all))
    se = _f(np.std(S_all, ddof=1)) / np.sqrt(n)
    t = stats.t.ppf(1 - alpha / 2, n - 1)
    return {"est": est, "lo": est - t * se, "hi": est + t * se, "extras": {}}


def arm_classical(S_all, Y_all, labeled_idx, alpha, seed, weights=None):
    """(2) classical_mean_ci on the m labeled oracle labels."""
    _check_alpha(alpha)
    Y_lab, _, _, _ = _split(S_all, Y_all, labeled_idx)
    lo, hi = classical_mean_ci(Y_lab, alpha=alpha)
    return {"est": _f(np.mean(Y_lab)), "lo": _f(lo), "hi": _f(hi), "extras": {}}


def arm_ppi_vanilla(S_all, Y_all, labeled_idx, alpha, seed, weights=None):
    """(3) PPI lam=1; Yhat = raw judge scores; disjoint labeled/unlabeled."""
    _check_alpha(alpha)
    Y_lab, S_lab, S_unl, _ = _split(S_all, Y_all, labeled_idx)
    est = _f(ppi_mean_pointestimate(Y_lab, S_lab, S_unl, lam=1))
    lo, hi = ppi_mean_ci(Y_lab, S_lab, S_unl, alpha=alpha, lam=1)
    return {"est": est, "lo": _f(lo), "hi": _f(hi), "extras": {}}


def arm_ppi_pp(S_all, Y_all, labeled_idx, alpha, seed, weights=None):
    """(4) PPI++ lam=None (power-tuned); records lam_hat."""
    _check_alpha(alpha)
    Y_lab, S_lab, S_unl, _ = _split(S_all, Y_all, labeled_idx)
    lam_hat = lam_hat_ppi(Y_lab, S_lab, S_unl)
    est = _f(ppi_mean_pointestimate(Y_lab, S_lab, S_unl, lam=None))
    lo, hi = ppi_mean_ci(Y_lab, S_lab, S_unl, alpha=alpha, lam=None)
    return {"est": est, "lo": _f(lo), "hi": _f(hi), "extras": {"lam_hat": lam_hat}}


def arm_ppi_pp_t(S_all, Y_all, labeled_idx, alpha, seed, weights=None):
    """(5) PPI++ with the half-width rescaled z -> t_{m-1} (quantile family)."""
    _check_alpha(alpha)
    base = arm_ppi_pp(S_all, Y_all, labeled_idx, alpha, seed, weights)
    m = len(labeled_idx)
    center = (base["lo"] + base["hi"]) / 2.0
    half = (base["hi"] - base["lo"]) / 2.0
    scale = stats.t.ppf(1 - alpha / 2, m - 1) / stats.norm.ppf(1 - alpha / 2)
    return {
        "est": base["est"],
        "lo": center - half * scale,
        "hi": center + half * scale,
        "extras": {"lam_hat": base["extras"]["lam_hat"], "t_scale": _f(scale)},
    }


def arm_postpi(S_all, Y_all, labeled_idx, alpha, seed, weights=None):
    """(6) PostPI (postprediction_ols_ci) with X = ones(N, 1) (mean via OLS).

    Included only because it passed validate.py's synthetic mean-recovery
    check. Uses numpy's GLOBAL RNG internally -> we seed it per call.
    """
    _check_alpha(alpha)
    Y_lab, S_lab, S_unl, _ = _split(S_all, Y_all, labeled_idx)
    np.random.seed(seed % (2**32))
    X_unl = np.ones((len(S_unl), 1))
    lo, hi = postprediction_ols_ci(Y_lab, S_lab, X_unl, S_unl, alpha=alpha)
    lo, hi = _f(lo), _f(hi)
    return {"est": (lo + hi) / 2.0, "lo": lo, "hi": hi, "extras": {}}


def arm_ppi_pp_weighted(S_all, Y_all, labeled_idx, alpha, seed, weights=None):
    """(9) PPI++ with inverse-inclusion-probability weights (stratified only).

    w_i = 1/pi_i on labeled rows; w_j = 1/(1 - pi_j) on the unlabeled
    remainder (its inclusion prob in the unlabeled set is 1 - pi_j).
    ppi_py internally normalizes weights to mean 1 (Hajek).
    """
    _check_alpha(alpha)
    assert weights is not None and "pi" in weights, "weighted arm needs pi"
    pi = np.asarray(weights["pi"], dtype=np.float64)
    Y_lab, S_lab, S_unl, mask = _split(S_all, Y_all, labeled_idx)
    pi_lab = pi[mask]
    pi_unl = pi[~mask]
    assert np.all(pi_lab > 0) and np.all(pi_unl < 1)
    w_lab = 1.0 / pi_lab
    w_unl = 1.0 / (1.0 - pi_unl)
    lam_hat = lam_hat_ppi(Y_lab, S_lab, S_unl, w=w_lab, w_unlabeled=w_unl)
    est = _f(
        ppi_mean_pointestimate(Y_lab, S_lab, S_unl, lam=None, w=w_lab, w_unlabeled=w_unl)
    )
    lo, hi = ppi_mean_ci(
        Y_lab, S_lab, S_unl, alpha=alpha, lam=None, w=w_lab, w_unlabeled=w_unl
    )
    return {"est": est, "lo": _f(lo), "hi": _f(hi), "extras": {"lam_hat": lam_hat}}


# ---------------------------------------------------------------------------
# Arms 7-8 (CJE-based; share one calibrated_mean_ci fit per replicate)
# ---------------------------------------------------------------------------


def fit_cje(S_all, Y_all, labeled_idx, alpha, seed):
    """(8) cje.calibrated_mean_ci on all n rows with NaN-masked oracle labels."""
    _check_alpha(alpha)
    n = len(S_all)
    Y_nan = np.full(n, np.nan)
    Y_nan[labeled_idx] = Y_all[labeled_idx]
    result = calibrated_mean_ci(S_all, Y_nan, alpha=alpha, seed=int(seed))
    return result


def cje_result_to_row(result) -> Dict[str, Any]:
    extras = {
        "inference_resolved": result.method,
        "selected_mode": result.calibrator.selected_mode,
    }
    boot = result.diagnostics.get("bootstrap")
    if boot:
        extras["bootstrap_skip_rate"] = boot["skip_rate"]
        extras["n_valid_replicates"] = boot["n_valid_replicates"]
    return {
        "est": _f(result.estimate),
        "lo": _f(result.ci[0]),
        "hi": _f(result.ci[1]),
        "extras": extras,
    }


def crossppi_from_calibrator(
    calibrator, S_all, Y_all, labeled_idx, alpha, seed
) -> Dict[str, Any]:
    """(7) cross-PPI fed with CJE's own cross-fitted isotonic fold models.

    Yhat (labeled) = out-of-fold predictions (each labeled row predicted by
    the model NOT trained on its fold). Yhat_unlabeled = (N, K) matrix, one
    column per fitted fold model. Fold identity validated in validate.py (d).
    """
    _check_alpha(alpha)
    n = len(S_all)
    Y_lab, _, S_unl, mask = _split(S_all, Y_all, labeled_idx)

    fold_ids = np.asarray(calibrator._fold_ids)
    # Cross-check the private attribute against the public fold function.
    recomputed = get_folds_for_prompts(
        [f"row_{i}" for i in range(n)], calibrator._n_folds, calibrator.random_seed
    )
    assert np.array_equal(fold_ids, recomputed), (
        "calibrator._fold_ids does not match the public "
        "get_folds_for_prompts recomputation — fold extraction is unsafe"
    )

    fitted = calibrator.get_fold_models_for_oua()
    fold_keys = sorted(fitted.keys())
    assert fold_keys, "no fitted fold models on the calibrator"

    # Labeled rows: proper OOF (fold ids are theirs; all labeled folds fitted).
    Yhat_oof = calibrator.predict_oof(S_all[mask], fold_ids[mask])
    # Unlabeled rows: (N, K) — column k = predictions from model f^(-k).
    Yhat_unl_2d = np.column_stack(
        [
            calibrator.predict_oof(S_unl, np.full(len(S_unl), k, dtype=int))
            for k in fold_keys
        ]
    )
    est = _f(crossppi_mean_pointestimate(Y_lab, Yhat_oof, Yhat_unl_2d))
    lo, hi = crossppi_mean_ci(Y_lab, Yhat_oof, Yhat_unl_2d, alpha=alpha)
    return {
        "est": est,
        "lo": _f(lo),
        "hi": _f(hi),
        "extras": {"K": len(fold_keys), "selected_mode": calibrator.selected_mode},
    }


# ---------------------------------------------------------------------------
# Registry + per-replicate orchestrator
# ---------------------------------------------------------------------------

# name -> adapter; arms 7 & 8 are handled jointly in evaluate_replicate
SIMPLE_ARMS: Dict[str, Callable] = {
    "naive_judge_t": arm_naive_judge,
    "classical": arm_classical,
    "ppi_vanilla": arm_ppi_vanilla,
    "ppi_pp": arm_ppi_pp,
    "ppi_pp_t": arm_ppi_pp_t,
    "postpi_ols_ones": arm_postpi,
}
STRATIFIED_ONLY_ARMS: Dict[str, Callable] = {
    "ppi_pp_weighted": arm_ppi_pp_weighted,
}
CJE_ARMS = ["cje_calibrated", "cross_ppi_cje_folds"]

# Set False by validate.py's check (c) verdict if PostPI fails mean recovery.
POSTPI_INCLUDED = True


def arm_names(regime: str) -> List[str]:
    names = list(SIMPLE_ARMS)
    if not POSTPI_INCLUDED:
        names.remove("postpi_ols_ones")
    names += CJE_ARMS
    if regime == "stratified":
        names += list(STRATIFIED_ONLY_ARMS)
    return names


def evaluate_replicate(
    S_all: np.ndarray,
    Y_all: np.ndarray,
    labeled_idx: np.ndarray,
    regime: str,
    seeds: Dict[str, int],
    pi: Optional[np.ndarray] = None,
    alpha: float = ALPHA,
) -> List[Dict[str, Any]]:
    """Run every arm for one (cell, rep). Returns one row dict per arm.

    seeds: {"cje": int, "<arm>": int, ...} per-rep child seeds.
    Exceptions are caught PER ARM: est/lo/hi = NaN, status records the error.
    """
    rows = []
    weights = {"pi": pi} if pi is not None else None

    def _run(name: str, fn: Callable, *args) -> Dict[str, Any]:
        t0 = time.time()
        try:
            out = fn(*args)
            status = "ok"
        except Exception as e:  # record, never fabricate
            out = {"est": np.nan, "lo": np.nan, "hi": np.nan, "extras": {}}
            status = f"error:{type(e).__name__}:{e}"
        out["estimator"] = name
        out["status"] = status
        out["runtime_s"] = time.time() - t0
        return out

    for name, fn in SIMPLE_ARMS.items():
        if name == "postpi_ols_ones" and not POSTPI_INCLUDED:
            continue
        rows.append(
            _run(name, fn, S_all, Y_all, labeled_idx, alpha,
                 seeds.get(name, seeds["cje"]), weights)
        )

    # Arms 8 then 7 (cross-PPI reuses arm 8's fitted calibrator).
    t0 = time.time()
    try:
        result = fit_cje(S_all, Y_all, labeled_idx, alpha, seeds["cje"])
        row8 = cje_result_to_row(result)
        row8.update(estimator="cje_calibrated", status="ok",
                    runtime_s=time.time() - t0)
        rows.append(row8)
        rows.append(
            _run("cross_ppi_cje_folds", crossppi_from_calibrator,
                 result.calibrator, S_all, Y_all, labeled_idx, alpha,
                 seeds["cje"])
        )
    except Exception as e:
        for name in CJE_ARMS:
            rows.append({
                "estimator": name, "est": np.nan, "lo": np.nan, "hi": np.nan,
                "extras": {}, "status": f"error:{type(e).__name__}:{e}",
                "runtime_s": time.time() - t0,
            })

    if regime == "stratified":
        for name, fn in STRATIFIED_ONLY_ARMS.items():
            rows.append(
                _run(name, fn, S_all, Y_all, labeled_idx, alpha,
                     seeds.get(name, seeds["cje"]), weights)
            )
    return rows
