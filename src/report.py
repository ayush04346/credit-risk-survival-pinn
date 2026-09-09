"""
Assembly of the study's headline table.

Every notebook writes its own arm to ``results/<arm>.json`` through
:func:`src.evaluate.save_arm_results`. Nothing here recomputes a metric or
retypes a number: the master table is built by reading those files, so a figure
in the table can always be traced back to the notebook run that produced it, and
a stale arm shows up as a missing row rather than as a plausible-looking number
somebody copied across.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import RESULTS_DIR

#: Canonical arm order for the comparison table: classical first, then the
#: survival models in increasing order of structural constraint.
#:
#: Each classical model appears twice. The full-data row is the best that model
#: does on this problem; the "(300k)" row is fitted on the same stratified draw
#: the survival arms use. Without the second row, any classical lead over a
#: survival arm mixes a real difference in discrimination with a five-fold
#: advantage in training rows, and the table cannot say which it is.
ARM_ORDER = [
    "Logistic (independent horizons)",
    "Logistic (300k)",
    "XGBoost (independent horizons)",
    "XGBoost (300k)",
    "Cox PH",
    "Unconstrained NN",
    "Soft-penalty NN",
    "Monotone-architecture NN",
]

MASTER_COLUMNS = [
    "arm",
    "auc_12m", "auc_24m", "auc_36m",
    "c_harrell", "c_uno", "ibs_1_60m",
    "pct_points_violating", "pct_borrowers_violating", "max_survival_increase",
    "pct_borrowers_any_PD_inversion",
    "n_train", "n_test",
]


def read_arm_jsons(results_dir=None) -> list:
    """Load every per-arm payload written by ``save_arm_results``."""
    d = results_dir or RESULTS_DIR
    payloads = []
    for p in sorted(d.glob("*.json")):
        with open(p) as fh:
            payload = json.load(fh)
        if "metrics" in payload and "arm" in payload:
            payload["_file"] = p.name
            payloads.append(payload)
    return payloads


def build_master_table(results_dir=None, write=True, order=ARM_ORDER) -> pd.DataFrame:
    """
    One row per arm, generated from the per-arm JSONs.

    Missing arms are reported as missing rather than silently omitted, so an
    incomplete run is visible in the artefact itself.
    """
    payloads = read_arm_jsons(results_dir)
    rows = []
    for p in payloads:
        m = p.get("metrics", {})
        mono = p.get("monotonicity", {})
        notes = p.get("notes", {})
        rows.append({
            "arm": p["arm"],
            "auc_12m": m.get("auc_12m"),
            "auc_24m": m.get("auc_24m"),
            "auc_36m": m.get("auc_36m"),
            "c_harrell": m.get("c_harrell"),
            "c_uno": m.get("c_uno"),
            "ibs_1_60m": m.get("ibs_1_60m"),
            "pct_points_violating": mono.get("pct_points_violating"),
            "pct_borrowers_violating": mono.get("pct_borrowers_violating"),
            "max_survival_increase": mono.get("max_survival_increase"),
            "pct_borrowers_any_PD_inversion": m.get("pct_borrowers_any_PD_inversion"),
            "n_train": notes.get("n_train"),
            "n_test": notes.get("n_test"),
        })

    df = pd.DataFrame(rows, columns=MASTER_COLUMNS)

    if order:
        found = {a: i for i, a in enumerate(order)}
        df["_o"] = df["arm"].map(found).fillna(len(order) + 1)
        df = df.sort_values(["_o", "arm"]).drop(columns="_o").reset_index(drop=True)

    if write:
        RESULTS_DIR.mkdir(exist_ok=True)
        df.to_csv(RESULTS_DIR / "master_comparison.csv", index=False)
    return df


def missing_arms(df: pd.DataFrame, order=ARM_ORDER) -> list:
    """Arms expected in the study that have not written a results file."""
    return [a for a in order if a not in set(df["arm"])]


def format_master(df: pd.DataFrame) -> pd.DataFrame:
    """Round the master table for display without touching the stored CSV."""
    out = df.copy()
    for c in ["auc_12m", "auc_24m", "auc_36m", "c_harrell", "c_uno", "ibs_1_60m"]:
        if c in out:
            out[c] = out[c].astype(float).round(4)
    for c in ["pct_points_violating", "pct_borrowers_violating",
              "pct_borrowers_any_PD_inversion"]:
        if c in out:
            out[c] = out[c].astype(float).round(3)
    if "max_survival_increase" in out:
        out["max_survival_increase"] = out["max_survival_increase"].map(
            lambda v: 0.0 if v in (None, 0) or (isinstance(v, float) and v == 0.0)
            else float("%.2e" % v)
        )
    for c in ["n_train", "n_test"]:
        if c in out:
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{int(v):,}")
    return out


def plot_cost_of_structure(df: pd.DataFrame, ax=None):
    """
    The study's summary picture: discrimination against structural validity.

    Each arm is one point. The horizontal axis is what the arm is worth as a
    ranking model; the vertical axis is how often it emits a term structure that
    cannot happen. An arm in the bottom right is the goal.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5.2))

    d = df.dropna(subset=["c_harrell"]).copy()
    y = d["pct_borrowers_violating"].astype(float).to_numpy()
    x = d["c_harrell"].astype(float).to_numpy()
    # A log axis would drop the structurally-valid arms at exactly zero, so the
    # axis stays linear and zero is drawn on the baseline where it belongs.
    ax.scatter(x, y, s=90, zorder=3)
    for xi, yi, name in zip(x, y, d["arm"]):
        ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(6, 6),
                    fontsize=8)
    ax.axhline(0, color="seagreen", lw=1.2, ls="--")
    ax.set_xlabel("Harrell's C (discrimination)")
    ax.set_ylabel("% of borrowers with a monotonicity violation")
    ax.set_title("What a structurally valid PD term structure costs")
    ax.grid(alpha=0.3)
    return ax
