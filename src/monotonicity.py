"""
Monotonicity audit: the central measurement of this study.

A survival function must be non-increasing in time, equivalently the cumulative
hazard must be non-decreasing, equivalently the hazard must be non-negative:

    S(t) = exp(-Lambda(t)),   lambda(t) = dLambda/dt >= 0

A model that violates this emits a PD term structure where, for some borrower,
the probability of having defaulted by 24 months is lower than by 12 months.
That is not a small numerical wrinkle: it is a structurally invalid term
structure, and it is the thing an IFRS 9 or Basel III staging rule would trip on.

This module measures how often that happens. It takes any arm that can produce a
cumulative hazard on a grid, samples held-out borrowers, evaluates on a dense
half-month grid from 1 to 60 months, and reports:

* the share of (borrower, time) points where dLambda/dt < 0
* the share of borrowers with at least one violation anywhere on the grid
* the largest single violation

The same function runs for every arm, so the numbers are directly comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import MONO_GRID, SEED


def cumhaz_from_survival(predict_survival, eps: float = 1e-12):
    """
    Adapt a ``predict_survival(X, times) -> S`` callable into a cumulative-hazard
    callable, via ``Lambda = -log(S)``.

    This is a strictly monotone transform of S, so it neither creates nor hides
    violations: S increases exactly where Lambda decreases.
    """
    def cumhaz(X, times):
        S = np.asarray(predict_survival(X, times), dtype=float)
        return -np.log(np.clip(S, eps, 1.0))
    return cumhaz


def audit_monotonicity(cumhaz_fn, X, name="model", n_borrowers=1000,
                       grid=MONO_GRID, seed=SEED, return_detail=False):
    """
    Run the audit for one arm.

    Parameters
    ----------
    cumhaz_fn : callable ``(X, times) -> Lambda`` of shape ``(len(X), len(times))``.
    X : held-out design matrix; ``n_borrowers`` rows are sampled from it.
    n_borrowers : number of held-out borrowers to audit (default 1000).
    grid : evaluation times in months (default 1 to 60 in 0.5-month steps).

    Returns
    -------
    dict of summary statistics, or ``(summary, detail)`` when ``return_detail``.
    ``detail`` carries the raw ``Lambda`` and ``dLambda/dt`` arrays for plotting.
    """
    grid = np.asarray(grid, float)
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.choice(n, size=min(n_borrowers, n), replace=False)

    Lam = np.asarray(cumhaz_fn(X[idx], grid), dtype=float)
    if Lam.shape != (len(idx), len(grid)):
        raise ValueError(
            "cumhaz_fn returned %s, expected %s" % (Lam.shape, (len(idx), len(grid)))
        )

    dt = np.diff(grid)[None, :]
    dLam = np.diff(Lam, axis=1)          # change in cumulative hazard
    deriv = dLam / dt                    # dLambda/dt, per month

    viol = deriv < 0
    n_points = viol.size
    n_viol = int(viol.sum())
    per_borrower = viol.any(axis=1)

    # Magnitude: the most negative derivative seen anywhere, and the largest
    # single backward step in Lambda that produced it.
    if n_viol:
        worst_deriv = float(-deriv.min())
        worst_drop = float(-dLam.min())
        # Same quantity expressed as a survival probability that goes the wrong
        # way: the largest increase in S between two adjacent grid points.
        S = np.exp(-Lam)
        worst_S_rise = float(np.diff(S, axis=1).max())
    else:
        worst_deriv = worst_drop = worst_S_rise = 0.0

    summary = {
        "arm": name,
        "n_borrowers": int(len(idx)),
        "n_grid_points": int(len(grid)),
        "pct_points_violating": 100.0 * n_viol / n_points,
        "pct_borrowers_violating": 100.0 * float(per_borrower.mean()),
        "max_violation_dLam_dt": worst_deriv,
        "max_violation_dLam": worst_drop,
        "max_survival_increase": worst_S_rise,
    }

    if return_detail:
        detail = {"idx": idx, "grid": grid, "Lambda": Lam, "deriv": deriv,
                  "violating_borrower": per_borrower}
        return summary, detail
    return summary


def audit_table(summaries) -> pd.DataFrame:
    """Stack per-arm summaries into the study's headline table."""
    df = pd.DataFrame(list(summaries))
    cols = ["arm", "n_borrowers", "n_grid_points", "pct_points_violating",
            "pct_borrowers_violating", "max_violation_dLam_dt",
            "max_violation_dLam", "max_survival_increase"]
    return df[[c for c in cols if c in df.columns]]


def format_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Round the audit table for display."""
    out = df.copy()
    for c in ["pct_points_violating", "pct_borrowers_violating"]:
        if c in out:
            out[c] = out[c].round(3)
    for c in ["max_violation_dLam_dt", "max_violation_dLam", "max_survival_increase"]:
        if c in out:
            out[c] = out[c].map(lambda v: float("%.3e" % v))
    return out


def plot_worst_curves(detail, n_curves: int = 5, name: str = "model", ax=None):
    """
    Plot the survival curves of the borrowers with the largest violations, so the
    failure is visible rather than only tabulated.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    grid = detail["grid"]
    Lam = detail["Lambda"]
    S = np.exp(-Lam)
    worst = np.argsort(np.diff(S, axis=1).max(axis=1))[::-1][:n_curves]

    for i in worst:
        rise = float(np.diff(S[i]).max())
        ax.plot(grid, S[i], lw=1.4,
                label="borrower %d (max rise %.2e)" % (int(i), rise))
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("S(t)")
    ax.set_title("Largest monotonicity violations - %s" % name)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    return ax
