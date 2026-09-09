"""
Shared PyTorch machinery for the neural arms.

Both network arms are trained by the same routine here, so the only difference
between them is the loss function each supplies. That matters: if one arm were
trained with a different optimiser schedule or a different stopping rule, any
difference in its metrics would be unattributable.

What this replaces
------------------
The earlier notebooks took **one full-batch Adam step per epoch** over roughly
1.57M training rows: 20 steps in notebook 03, 30 in notebook 04. Twenty gradient
steps is not training, and the stored loss trace showed it -- 0.5800, 0.4101,
0.4129, 0.4133, improving for five steps and then drifting upwards. There was
also no validation split and no stopping rule.

:func:`train_minibatch` does proper minibatch SGD: shuffled batches of at least
4096, real passes over the data, a validation slice held out of TRAIN, early
stopping on validation loss with best-weight restoration, and a recorded loss
history for plotting.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .common import SEED


def set_seed(seed: int = SEED):
    """Seed every source of randomness the neural arms touch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(False)
    return seed


def as_tensors(X, t, e, device="cpu"):
    """Design matrix, observed time and event indicator as column tensors."""
    X = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
    t = torch.as_tensor(np.asarray(t, dtype=np.float32).reshape(-1, 1), device=device)
    e = torch.as_tensor(np.asarray(e, dtype=np.float32).reshape(-1, 1), device=device)
    return X, t, e


def train_minibatch(model, step_loss, X, t, e, *, batch_size=8192, max_epochs=40,
                    lr=1e-3, val_frac=0.1, patience=5, seed=SEED, monitor="data",
                    device="cpu", verbose=True):
    """
    Minibatch Adam with a validation split and early stopping.

    Parameters
    ----------
    step_loss : callable ``(model, xb, tb, eb) -> (total_loss, parts)``.
        ``parts`` is a dict of named scalar components; ``monitor`` selects the
        one early stopping watches. The physics penalty is a regulariser rather
        than a measure of fit, so the default watches the data likelihood.
    val_frac : fraction of TRAIN held out for the stopping rule. The test split
        is never touched here.
    patience : epochs without validation improvement before stopping. The best
        weights seen are restored before returning.

    Returns
    -------
    history : dict of per-epoch lists, ready to plot.
    """
    g = torch.Generator().manual_seed(seed)
    n = len(X)
    perm = torch.randperm(n, generator=g)
    n_val = int(val_frac * n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    Xtr, ttr, etr = X[tr_idx], t[tr_idx], e[tr_idx]
    Xva, tva, eva = X[val_idx], t[val_idx], e[val_idx]

    loader = DataLoader(TensorDataset(Xtr, ttr, etr), batch_size=batch_size,
                        shuffle=True, generator=g, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"epoch": [], "train_total": [], "val_total": [], "val_monitor": [],
               "steps": [], "seconds": []}
    part_names = None

    best = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch, bad, total_steps = 0, 0, 0
    t0 = time.time()

    if verbose:
        print(f"train rows {len(Xtr):,} | val rows {len(Xva):,} | "
              f"batch {batch_size} | {int(np.ceil(len(Xtr) / batch_size))} steps/epoch")

    for epoch in range(1, max_epochs + 1):
        model.train()
        run, seen = 0.0, 0
        run_parts = {}
        for xb, tb, eb in loader:
            opt.zero_grad(set_to_none=True)
            loss, parts = step_loss(model, xb, tb, eb)
            loss.backward()
            opt.step()
            total_steps += 1
            bs = len(xb)
            run += float(loss.detach()) * bs
            seen += bs
            for k, v in parts.items():
                run_parts[k] = run_parts.get(k, 0.0) + float(v) * bs

        train_total = run / seen
        train_parts = {k: v / seen for k, v in run_parts.items()}

        # Validation. Grad stays enabled because arms whose hazard is an autograd
        # derivative in t cannot compute their loss under no_grad.
        model.eval()
        v_tot, v_parts, v_seen = 0.0, {}, 0
        for s in range(0, len(Xva), batch_size):
            xb, tb, eb = Xva[s:s + batch_size], tva[s:s + batch_size], eva[s:s + batch_size]
            loss, parts = step_loss(model, xb, tb, eb)
            bs = len(xb)
            v_tot += float(loss.detach()) * bs
            v_seen += bs
            for k, v in parts.items():
                v_parts[k] = v_parts.get(k, 0.0) + float(v) * bs
        val_total = v_tot / v_seen
        val_parts = {k: v / v_seen for k, v in v_parts.items()}
        watch = val_parts.get(monitor, val_total)

        if part_names is None:
            part_names = sorted(train_parts)
            for nm in part_names:
                history["train_" + nm] = []
                history["val_" + nm] = []

        history["epoch"].append(epoch)
        history["train_total"].append(train_total)
        history["val_total"].append(val_total)
        history["val_monitor"].append(watch)
        history["steps"].append(total_steps)
        history["seconds"].append(time.time() - t0)
        for nm in part_names:
            history["train_" + nm].append(train_parts.get(nm, float("nan")))
            history["val_" + nm].append(val_parts.get(nm, float("nan")))

        improved = watch < best - 1e-6
        if improved:
            best, best_epoch, bad = watch, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1

        if verbose:
            extra = "  ".join(f"{k}={v:.5f}" for k, v in sorted(val_parts.items()))
            print(f"epoch {epoch:>3} | train {train_total:.5f} | val {val_total:.5f} | "
                  f"{extra} | {'*' if improved else ' '} ({time.time() - t0:.0f}s)")

        if bad >= patience:
            if verbose:
                print(f"early stop at epoch {epoch}; best epoch {best_epoch} "
                      f"({monitor}={best:.5f})")
            break

    model.load_state_dict(best_state)
    model.eval()
    history["best_epoch"] = best_epoch
    history["best_monitor"] = best
    history["monitor"] = monitor
    history["total_steps"] = total_steps
    if verbose:
        print(f"restored weights from epoch {best_epoch} | {total_steps} gradient steps "
              f"in {time.time() - t0:.0f}s")
    return history


def plot_history(history, title="training", ax=None):
    """Train and validation loss against epoch."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.2))
    ep = history["epoch"]
    ax.plot(ep, history["train_total"], "o-", ms=3, label="train (total)")
    ax.plot(ep, history["val_total"], "o-", ms=3, label="val (total)")
    if "val_data" in history and history["val_data"][0] == history["val_data"][0]:
        ax.plot(ep, history["val_data"], "s--", ms=3, label="val (data NLL)")
    be = history.get("best_epoch")
    if be:
        ax.axvline(be, color="grey", ls=":", lw=1.2, label=f"best epoch {be}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def batched_survival(surv_fn, batch=100_000, device="cpu", needs_grad=False):
    """
    Wrap a torch model as the ``predict_survival(X, times) -> S`` callable that
    :mod:`src.evaluate` and :mod:`src.monotonicity` both expect.

    ``surv_fn(x, t)`` takes two tensors of matching length and returns S. Rows are
    processed in chunks so that a 674k-row test set crossed with a 60-point time
    grid never materialises as one tensor.
    """
    def predict_survival(X, times):
        times = np.atleast_1d(np.asarray(times, dtype=np.float32))
        X = np.asarray(X, dtype=np.float32)
        out = np.empty((len(X), len(times)), dtype=np.float64)
        ctx = torch.enable_grad() if needs_grad else torch.no_grad()
        with ctx:
            for s in range(0, len(X), batch):
                xb = torch.as_tensor(X[s:s + batch], device=device)
                for k, tv in enumerate(times):
                    tb = torch.full((len(xb), 1), float(tv), device=device)
                    S = surv_fn(xb, tb)
                    out[s:s + batch, k] = S.detach().cpu().numpy().ravel()
        return out
    return predict_survival


def batched_cumhaz(cumhaz_fn, batch=100_000, device="cpu"):
    """Same wrapper for an arm that exposes cumulative hazard directly."""
    def predict_cumhaz(X, times):
        times = np.atleast_1d(np.asarray(times, dtype=np.float32))
        X = np.asarray(X, dtype=np.float32)
        out = np.empty((len(X), len(times)), dtype=np.float64)
        with torch.no_grad():
            for s in range(0, len(X), batch):
                xb = torch.as_tensor(X[s:s + batch], device=device)
                for k, tv in enumerate(times):
                    tb = torch.full((len(xb), 1), float(tv), device=device)
                    out[s:s + batch, k] = cumhaz_fn(xb, tb).detach().cpu().numpy().ravel()
        return out
    return predict_cumhaz
