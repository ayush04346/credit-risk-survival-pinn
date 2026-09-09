"""
Shared data, split and feature-construction logic for every model arm.

This module is the single source of truth. No notebook re-implements loading,
splitting or feature building: every arm calls ``load_data`` -> ``make_split``
-> ``build_features`` so that all models are fitted and scored on identical
matrices, and the only thing that varies between arms is the model itself.

Design rule enforced here
-------------------------
Every transformation that has to *learn* something from the data (the standard
scaler's means and variances, the one-hot encoder's category list, the median
used to impute ``emp_length``) is fitted on the TRAIN split only and then
applied to test. Notebook 04 previously standardised the full design matrix
before splitting, which leaked test-set moments into training. ``build_features``
removes that leak for all arms at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RAW_GZ = DATA_DIR / "accepted_2007_to_2018Q4.csv.gz"
CLEAN_CSV = DATA_DIR / "lendingclub_survival_clean.csv"

# --------------------------------------------------------------------------
# Protocol constants shared by every arm
# --------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.30
HORIZONS = (12, 24, 36)

#: Training-set subsample used by the arms whose fitting cost is superlinear or
#: otherwise prohibitive on the full 1.57M training rows (the neural arms and
#: Cox). The TEST split is never subsampled, so every arm is scored on the same
#: 674,272 held-out loans and the metrics stay comparable across the whole table.
#: Each arm records its own ``n_train`` in its results JSON, so where an arm sits
#: is always visible rather than assumed.
SUBSAMPLE_N = 300_000

#: Evaluation grid for the integrated Brier score (months).
IBS_GRID = np.arange(1.0, 61.0, 1.0)

#: Dense grid for the monotonicity audit (months, half-month steps).
MONO_GRID = np.arange(1.0, 60.0 + 1e-9, 0.5)

#: Loan statuses treated as a default event.
DEFAULT_STATUSES = ["Charged Off", "Default", "Late (31-120 days)"]

#: Columns pulled out of the 151-column raw Kaggle extract.
RAW_COLUMNS = [
    "loan_status", "issue_d", "last_pymnt_d", "term", "annual_inc",
    "emp_length", "int_rate", "loan_amnt", "dti", "grade", "home_ownership",
]

# --------------------------------------------------------------------------
# Feature specification
# --------------------------------------------------------------------------
NUMERIC_FEATURES = ["annual_inc", "dti", "loan_amnt", "int_rate", "term"]
ORDINAL_FEATURES = ["grade", "emp_length"]
CATEGORICAL_FEATURES = ["home_ownership"]

GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]
GRADE_MAP = {g: i for i, g in enumerate(GRADE_ORDER)}

EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}


@dataclass
class FeatureBundle:
    """Fitted design matrices plus the transformers used to build them."""

    X_train: np.ndarray
    X_test: np.ndarray
    names: list
    scaler: StandardScaler
    encoder: OneHotEncoder
    emp_length_median: float

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the already-fitted pipeline to a fresh frame."""
        raw = _raw_design(df, self.emp_length_median)
        cat = self.encoder.transform(df[CATEGORICAL_FEATURES].astype(str))
        return self.scaler.transform(np.hstack([raw, cat])).astype(np.float32)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_data(path=None) -> pd.DataFrame:
    """Read the cleaned survival dataset written by notebook 01."""
    path = Path(path) if path is not None else CLEAN_CSV
    if not path.exists():
        raise FileNotFoundError(
            str(path) + " not found. Run Notebooks/01_lendingclub_data_preparation.ipynb "
            "first (it needs the Kaggle accepted-loans extract in data/)."
        )
    return pd.read_csv(path, low_memory=False)


def survival_arrays(df: pd.DataFrame):
    """Return ``(time, event)`` as numpy arrays."""
    return df["time"].to_numpy(np.float64), df["event"].to_numpy(np.int64)


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def make_split(seed: int = SEED, df=None, test_size: float = TEST_SIZE):
    """
    Produce THE train/test split. Every arm calls this with the same seed, so
    all models are compared on identical rows.

    Returns ``(train_df, test_df)`` with the original index preserved.
    """
    if df is None:
        df = load_data()
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, shuffle=True
    )
    return train_df, test_df


def subsample_train(train_df: pd.DataFrame, n: int = SUBSAMPLE_N, seed: int = SEED,
                    stratify_on: str = "event") -> pd.DataFrame:
    """
    Stratified subsample of the TRAINING split only.

    Fitting some arms on all 1.57M training rows does not finish in acceptable
    wall-clock on this machine. Rather than leave those arms unmeasured, they are
    fitted on a stratified draw that preserves the event rate exactly, and every
    arm records the ``n_train`` it actually used.

    The test split is deliberately not touched: all arms are scored on the same
    674,272 held-out loans, so the comparison across the table stays valid even
    where the training sizes differ.

    Passing ``n=None`` or an ``n`` at least as large as the frame returns it
    unchanged, so an arm can opt out by asking for the full set.
    """
    if n is None or n >= len(train_df):
        return train_df

    rng = np.random.default_rng(seed)
    parts = []
    groups = train_df.groupby(stratify_on, sort=True)
    for _, g in groups:
        take = int(round(n * len(g) / len(train_df)))
        take = max(1, min(take, len(g)))
        parts.append(g.iloc[rng.choice(len(g), size=take, replace=False)])
    out = pd.concat(parts).sample(frac=1.0, random_state=seed)
    return out


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def _raw_design(df: pd.DataFrame, emp_median: float) -> np.ndarray:
    """Numeric + ordinal block, before scaling. Nothing is fitted here."""
    out = pd.DataFrame(index=df.index)
    for c in NUMERIC_FEATURES:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    out["grade"] = df["grade"].map(GRADE_MAP).astype(float)

    emp = df["emp_length"].map(EMP_LENGTH_MAP).astype(float)
    out["emp_length_missing"] = emp.isna().astype(float)
    out["emp_length"] = emp.fillna(emp_median)

    return out.to_numpy(float)


def _raw_design_names() -> list:
    return NUMERIC_FEATURES + ["grade", "emp_length_missing", "emp_length"]


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> FeatureBundle:
    """
    Build the shared design matrix.

    Feature set: ``annual_inc``, ``dti``, ``loan_amnt``, ``int_rate``, ``term``
    numeric; ``grade`` ordinal A..G -> 0..6; ``emp_length`` ordinal 0..10 with a
    missing indicator; ``home_ownership`` one-hot.

    Everything is fitted on ``train_df`` only. The scaler never sees test rows.
    """
    emp_median = float(
        train_df["emp_length"].map(EMP_LENGTH_MAP).astype(float).median()
    )

    raw_tr = _raw_design(train_df, emp_median)
    raw_te = _raw_design(test_df, emp_median)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)
    encoder.fit(train_df[CATEGORICAL_FEATURES].astype(str))
    cat_tr = encoder.transform(train_df[CATEGORICAL_FEATURES].astype(str))
    cat_te = encoder.transform(test_df[CATEGORICAL_FEATURES].astype(str))

    Xtr = np.hstack([raw_tr, cat_tr])
    Xte = np.hstack([raw_te, cat_te])

    if np.isnan(Xtr).any() or np.isnan(Xte).any():
        raise ValueError("NaNs present in design matrix after encoding")

    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)

    names = _raw_design_names() + list(
        encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    )
    return FeatureBundle(Xtr, Xte, names, scaler, encoder, emp_median)


def horizon_labels(df: pd.DataFrame, horizons=HORIZONS) -> dict:
    """
    Binary default-by-H labels, defined exactly as notebook 02 defines them:
    ``y_H = (time <= H) & (event == 1)``.

    Keeping one definition here is what makes the survival arms' time-dependent
    AUCs directly comparable to the classical baselines.
    """
    t = df["time"].to_numpy()
    e = df["event"].to_numpy()
    return {H: ((t <= H) & (e == 1)).astype(int) for H in horizons}


def describe_split(train_df, test_df) -> pd.DataFrame:
    """Small sanity table printed by every notebook."""
    rows = []
    for name, d in (("train", train_df), ("test", test_df)):
        rows.append({
            "split": name,
            "rows": len(d),
            "event_rate": round(float(d["event"].mean()), 6),
            "median_time_m": round(float(d["time"].median()), 3),
        })
    return pd.DataFrame(rows)


def notebook_setup():
    """
    Put the repository root on ``sys.path`` so notebooks in ``Notebooks/`` can
    ``from src.common import ...``. Safe to call repeatedly.
    """
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return ROOT
