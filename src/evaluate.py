"""
Evaluation harness shared by every model arm.

One entry point, :func:`evaluate_arm`, takes a fitted model's predicted survival
function and returns discrimination, calibration and accuracy metrics on the
held-out test set:

* Harrell's C-index, and Uno's inverse-probability-of-censoring-weighted C.
* Time-dependent AUC at 12, 24 and 36 months, computed exactly the way notebook
  02 computes its binary AUCs, so survival arms and classical baselines land on
  the same scale.
* Integrated Brier score over 1-60 months, IPCW-corrected for censoring.
* A decile calibration curve at 12, 24 and 36 months, where the observed default
  rate inside each bin is a Kaplan-Meier estimate rather than a raw mean, so
  censored loans do not bias it downwards.

The model is supplied as a callable ``predict_survival(X, times) -> S`` where
``S`` has shape ``(len(X), len(times))`` and ``S[i, k]`` is the predicted
probability that borrower ``i`` has not defaulted by ``times[k]``. Any arm that
can produce that array can be scored here, which is what makes the comparison
apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .common import HORIZONS, IBS_GRID, SEED

# --------------------------------------------------------------------------
# Optional backends
# --------------------------------------------------------------------------
try:  # pragma: no cover - environment dependent
    from lifelines import KaplanMeierFitter
    from lifelines.utils import concordance_index as _lifelines_cindex
    HAVE_LIFELINES = True
except Exception:  # pragma: no cover
    HAVE_LIFELINES = False

try:  # pragma: no cover - environment dependent
    from sksurv.metrics import concordance_index_ipcw as _sksurv_uno
    HAVE_SKSURV = True
except Exception:  # pragma: no cover
    HAVE_SKSURV = False


def available_backends() -> dict:
    return {"lifelines": HAVE_LIFELINES, "scikit-survival": HAVE_SKSURV}


# --------------------------------------------------------------------------
# Censoring distribution
# --------------------------------------------------------------------------
class CensoringKM:
    """
    Kaplan-Meier estimate of the censoring survival function G(t) = P(C > t),
    fitted on the training split and used for all IPCW weights.

    Fitting G on train rather than test keeps the weights independent of the
    predictions being scored.
    """

    def __init__(self, time, event, floor: float = 1e-3):
        time = np.asarray(time, float)
        event = np.asarray(event, int)
        # The censoring indicator is the complement of the event indicator.
        order = np.argsort(time, kind="mergesort")
        t_sorted = time[order]
        cens = (1 - event)[order]

        uniq, start = np.unique(t_sorted, return_index=True)
        n = len(t_sorted)
        at_risk = n - start                      # number with T >= t
        d_cens = np.add.reduceat(cens, start)    # censorings at each unique t

        with np.errstate(divide="ignore", invalid="ignore"):
            factors = np.where(at_risk > 0, 1.0 - d_cens / at_risk, 1.0)
        self._t = uniq
        self._g = np.clip(np.cumprod(factors), floor, 1.0)
        self._floor = floor

    def __call__(self, t):
        """G evaluated just before t (left-continuous), clipped away from zero."""
        t = np.asarray(t, float)
        idx = np.searchsorted(self._t, t, side="right") - 1
        out = np.where(idx < 0, 1.0, self._g[np.clip(idx, 0, len(self._g) - 1)])
        return np.clip(out, self._floor, 1.0)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def integrated_survival_score(S: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Collapse a survival curve to one risk-ordering score.

    Uses the area under S(t) over the evaluation grid, i.e. restricted mean
    survival time. Higher means the borrower is predicted to survive longer.
    A single scalar is required because the C-index ranks borrowers, and RMST
    is the summary that does not privilege any one horizon.
    """
    return np.trapezoid(S, times, axis=1) if hasattr(np, "trapezoid") else np.trapz(S, times, axis=1)


def harrell_c_index(time, event, survival_score) -> float:
    """Harrell's C. ``survival_score`` is higher for longer predicted survival."""
    if HAVE_LIFELINES:
        return float(_lifelines_cindex(time, survival_score, event))
    return _pairwise_c(time, event, survival_score, weights=None)


def uno_c_index(time, event, survival_score, cens_km: CensoringKM,
                tau: float | None = None) -> float:
    """
    Uno's C: concordance weighted by ``1 / G(T_i)^2`` over comparable pairs,
    which removes the dependence on the study-specific censoring distribution
    that biases Harrell's C.
    """
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    w = 1.0 / np.square(cens_km(time))
    return _pairwise_c(time, event, survival_score, weights=w, tau=tau)


def _pairwise_c(time, event, survival_score, weights=None, tau=None,
                block: int = 256) -> float:
    """
    Weighted concordance over comparable pairs (T_i < T_j, delta_i = 1).
    Computed in blocks so the O(n^2) comparison never materialises at once.
    """
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    score = np.asarray(survival_score, float)
    if weights is None:
        weights = np.ones_like(time)

    valid = event == 1
    if tau is not None:
        valid &= time <= tau
    idx_i = np.flatnonzero(valid)
    if idx_i.size == 0:
        return float("nan")

    num = 0.0
    den = 0.0
    for s in range(0, idx_i.size, block):
        ii = idx_i[s:s + block]
        ti = time[ii][:, None]
        si = score[ii][:, None]
        wi = weights[ii][:, None]

        comparable = ti < time[None, :]
        if not comparable.any():
            continue
        # Shorter observed time should carry the lower survival score.
        concordant = (si < score[None, :]) & comparable
        tied = (si == score[None, :]) & comparable

        num += float((wi * (concordant + 0.5 * tied)).sum())
        den += float((wi * comparable).sum())

    return float(num / den) if den > 0 else float("nan")


def time_dependent_auc(S_at_h: np.ndarray, y: np.ndarray) -> float:
    """
    AUC at one horizon, defined exactly as notebook 02 defines it: the binary
    label is ``(time <= H) & (event == 1)`` over the whole test set, and the
    score is the predicted probability of defaulting by H, i.e. ``1 - S(H)``.
    """
    return float(roc_auc_score(y, 1.0 - S_at_h))


def brier_curve(S: np.ndarray, times: np.ndarray, time, event,
                cens_km: CensoringKM):
    """
    IPCW Brier score at each grid point (Graf et al. 1999).

    Loans that default on or before t contribute ``S(t)^2 / G(T)``; loans still
    alive at t contribute ``(1 - S(t))^2 / G(t)``; loans censored before t
    contribute nothing and are compensated for by the weights.
    """
    time = np.asarray(time, float)[:, None]
    event = np.asarray(event, int)[:, None]
    tg = np.asarray(times, float)[None, :]

    g_ti = cens_km(time.ravel())[:, None]
    g_t = cens_km(times)[None, :]

    died_before = (time <= tg) & (event == 1)
    alive_after = time > tg

    contrib = np.where(died_before, np.square(S) / g_ti, 0.0)
    contrib += np.where(alive_after, np.square(1.0 - S) / g_t, 0.0)
    return contrib.mean(axis=0)


def integrated_brier_score(S, times, time, event, cens_km) -> float:
    bs = brier_curve(S, times, time, event, cens_km)
    times = np.asarray(times, float)
    area = np.trapezoid(bs, times) if hasattr(np, "trapezoid") else np.trapz(bs, times)
    return float(area / (times[-1] - times[0]))


def calibration_at_horizon(pd_hat: np.ndarray, time, event, horizon: float,
                           n_bins: int = 10) -> pd.DataFrame:
    """
    Decile calibration at one horizon.

    Predicted PD is the mean of ``1 - S(H)`` in the bin. Observed PD is
    ``1 - KM(H)`` inside the bin, so loans censored before H do not silently
    depress the observed rate the way a raw mean would.
    """
    pd_hat = np.asarray(pd_hat, float)
    time = np.asarray(time, float)
    event = np.asarray(event, int)

    edges = np.quantile(pd_hat, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    bins = np.digitize(pd_hat, edges[1:-1], right=True)

    rows = []
    for b in range(len(edges) - 1):
        m = bins == b
        if m.sum() < 50:
            continue
        rows.append({
            "bin": b,
            "n": int(m.sum()),
            "predicted_pd": float(pd_hat[m].mean()),
            "observed_pd": _km_failure(time[m], event[m], horizon),
        })
    return pd.DataFrame(rows)


def _km_failure(time, event, horizon) -> float:
    """1 - S_KM(horizon) within a group."""
    if HAVE_LIFELINES:
        km = KaplanMeierFitter().fit(time, event)
        return float(1.0 - km.predict(horizon))
    order = np.argsort(time, kind="mergesort")
    t, e = np.asarray(time)[order], np.asarray(event)[order]
    n = len(t)
    surv = 1.0
    for i, (ti, ei) in enumerate(zip(t, e)):
        if ti > horizon:
            break
        if ei == 1:
            surv *= 1.0 - 1.0 / (n - i)
    return float(1.0 - surv)


# --------------------------------------------------------------------------
# Top-level harness
# --------------------------------------------------------------------------
@dataclass
class EvalResult:
    name: str
    metrics: dict
    calibration: dict = field(default_factory=dict)
    brier: pd.DataFrame = None
    notes: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([dict(arm=self.name, **self.metrics)])

    def __repr__(self):
        return "EvalResult(%s)\n%s" % (self.name, self.to_frame().to_string(index=False))


def evaluate_arm(predict_survival, X_test, test_df, train_df, name="model",
                 horizons=HORIZONS, ibs_grid=IBS_GRID, seed=SEED,
                 c_index_n=25_000, ibs_n=100_000) -> EvalResult:
    """
    Score one arm on the held-out test set.

    Parameters
    ----------
    predict_survival : callable ``(X, times) -> S`` of shape ``(len(X), len(times))``.
    X_test : design matrix from :func:`src.common.build_features`.
    test_df, train_df : the frames returned by :func:`src.common.make_split`.
    c_index_n, ibs_n : subsample sizes. The C-indices are pairwise and therefore
        quadratic, and the Brier score needs the full grid for every row, so both
        are computed on a fixed-seed subsample of the test set rather than all
        674k rows. Time-dependent AUC always uses the FULL test set, because that
        is the number being compared against notebook 02.
    """
    rng = np.random.default_rng(seed)
    t_te = test_df["time"].to_numpy(float)
    e_te = test_df["event"].to_numpy(int)
    cens = CensoringKM(train_df["time"].to_numpy(float),
                       train_df["event"].to_numpy(int))

    metrics = {}
    notes = {"n_test": int(len(test_df)), "backends": available_backends()}

    # ---- time-dependent AUC on the FULL test set -------------------------
    S_h = np.asarray(predict_survival(X_test, np.asarray(horizons, float)))
    for k, H in enumerate(horizons):
        y = ((t_te <= H) & (e_te == 1)).astype(int)
        metrics["auc_%dm" % H] = time_dependent_auc(S_h[:, k], y)

    # ---- C-indices on a fixed subsample ----------------------------------
    n = len(test_df)
    idx_c = rng.choice(n, size=min(c_index_n, n), replace=False)
    S_c = np.asarray(predict_survival(X_test[idx_c], ibs_grid))
    rmst = integrated_survival_score(S_c, ibs_grid)
    metrics["c_harrell"] = harrell_c_index(t_te[idx_c], e_te[idx_c], rmst)
    metrics["c_uno"] = uno_c_index(t_te[idx_c], e_te[idx_c], rmst, cens,
                                   tau=float(ibs_grid[-1]))
    notes["c_index_n"] = int(len(idx_c))

    # ---- integrated Brier score ------------------------------------------
    idx_b = rng.choice(n, size=min(ibs_n, n), replace=False)
    S_b = np.asarray(predict_survival(X_test[idx_b], ibs_grid))
    bs = brier_curve(S_b, ibs_grid, t_te[idx_b], e_te[idx_b], cens)
    area = np.trapezoid(bs, ibs_grid) if hasattr(np, "trapezoid") else np.trapz(bs, ibs_grid)
    metrics["ibs_1_60m"] = float(area / (ibs_grid[-1] - ibs_grid[0]))
    notes["ibs_n"] = int(len(idx_b))
    brier_df = pd.DataFrame({"month": ibs_grid, "brier": bs})

    # ---- calibration ------------------------------------------------------
    calib = {}
    for k, H in enumerate(horizons):
        calib[H] = calibration_at_horizon(1.0 - S_h[:, k], t_te, e_te, H)

    return EvalResult(name=name, metrics=metrics, calibration=calib,
                      brier=brier_df, notes=notes)


def comparison_table(results) -> pd.DataFrame:
    """Stack several :class:`EvalResult` objects into one table."""
    return pd.concat([r.to_frame() for r in results], ignore_index=True)


def plot_calibration(result: EvalResult, horizons=HORIZONS, ax=None):
    """Calibration curves at each horizon on one axis."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))
    for H in horizons:
        c = result.calibration[H]
        ax.plot(c["predicted_pd"], c["observed_pd"], "o-", label="%dm" % H)
    lim = max(
        float(result.calibration[H][["predicted_pd", "observed_pd"]].to_numpy().max())
        for H in horizons
    )
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="perfect")
    ax.set_xlabel("Predicted PD")
    ax.set_ylabel("Observed PD (Kaplan-Meier)")
    ax.set_title("Calibration - %s" % result.name)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


# --------------------------------------------------------------------------
# Adapters for arms that are not natively survival models
# --------------------------------------------------------------------------
def horizon_models_to_survival(models, horizons, predict_proba):
    """
    Wrap a set of independent per-horizon classifiers (notebook 02) as a
    ``predict_survival`` callable, so the classical baselines can be pushed
    through the same harness.

    S(H) is taken as ``1 - P(default by H)`` at the fitted horizons and linearly
    interpolated in between, with S(0) = 1. Nothing forces the result to be
    monotone: that is precisely the property being measured, so no sorting or
    clipping to a monotone envelope is applied.

    Beyond the last fitted horizon the curve is held **flat**. These models make
    no statement about month 37 onwards, and every way of inventing one is
    arbitrary; holding the last value constant is the choice that adds the least.
    Extrapolating the 24-to-36-month slope instead would drive predicted survival
    far below anything the model claims and would load 40% of the 1-60 month
    Brier integral with an artefact of the adapter rather than a property of the
    model. The limitation is real and belongs to the arm, but it should be
    reported as "undefined past 36 months", not smuggled in as a bad prediction.
    """
    hs = np.asarray(sorted(horizons), float)

    def predict_survival(X, times):
        cols = np.column_stack([1.0 - predict_proba(models[int(h)], X) for h in hs])
        anchor_t = np.concatenate([[0.0], hs])
        anchor_S = np.column_stack([np.ones(len(X)), cols])
        times = np.asarray(times, float)
        out = np.empty((len(X), len(times)))
        for k, t in enumerate(times):
            if t >= anchor_t[-1]:                      # flat past the last horizon
                out[:, k] = anchor_S[:, -1]
                continue
            j = np.searchsorted(anchor_t, t, side="right") - 1
            j = min(max(j, 0), len(anchor_t) - 2)
            t0, t1 = anchor_t[j], anchor_t[j + 1]
            w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            out[:, k] = anchor_S[:, j] + w * (anchor_S[:, j + 1] - anchor_S[:, j])
        return np.clip(out, 1e-8, 1.0)

    return predict_survival


# --------------------------------------------------------------------------
# Persistence, so the cross-notebook comparison table can be assembled
# --------------------------------------------------------------------------
def save_arm_results(result: EvalResult, mono: dict = None, extra: dict = None):
    """
    Write one arm's metrics to ``results/<arm>.json`` plus its calibration and
    Brier curves as CSV. Notebooks run independently, so the final comparison is
    assembled from these files rather than from in-memory state.
    """
    import json
    import re
    from .common import RESULTS_DIR

    RESULTS_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", result.name.lower()).strip("_")

    payload = {"arm": result.name, "metrics": result.metrics,
               "monotonicity": mono or {}, "notes": result.notes,
               "extra": extra or {}}
    with open(RESULTS_DIR / (slug + ".json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=float)

    if result.brier is not None:
        result.brier.to_csv(RESULTS_DIR / (slug + "_brier.csv"), index=False)
    for H, c in (result.calibration or {}).items():
        c.to_csv(RESULTS_DIR / (slug + "_calibration_%dm.csv" % H), index=False)
    return RESULTS_DIR / (slug + ".json")


def load_all_results(results_dir=None) -> pd.DataFrame:
    """Read every ``results/*.json`` back into one comparison table."""
    import json
    from .common import RESULTS_DIR

    d = results_dir or RESULTS_DIR
    rows = []
    for p in sorted(d.glob("*.json")):
        with open(p) as fh:
            payload = json.load(fh)
        if "metrics" not in payload:
            continue
        rows.append(dict(arm=payload["arm"], **payload["metrics"]))
    return pd.DataFrame(rows)


def load_all_monotonicity(results_dir=None) -> pd.DataFrame:
    """Read every arm's monotonicity audit back into one table."""
    import json
    from .common import RESULTS_DIR

    d = results_dir or RESULTS_DIR
    rows = []
    for p in sorted(d.glob("*.json")):
        with open(p) as fh:
            payload = json.load(fh)
        if payload.get("monotonicity"):
            rows.append(payload["monotonicity"])
    return pd.DataFrame(rows)
