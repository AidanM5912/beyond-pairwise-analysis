#!/usr/bin/env python3
"""
03_fit_ising_pl.py

Fit a pairwise Ising model per window using node-wise logistic
*pseudolikelihood* with L2- or elastic-net regularization. Outputs
(J, h) for each window plus pseudo-log-loss diagnostics.

Inputs:
  counts .npz   (from 02_make_windows.py) -> X [n_units, n_bins], bin_ms, duration_ms
  windows .tsv  (from 02_make_windows.py) -> start_bin, end_bin

Outputs:
  <out_root>/ising.npz
    - J_list             : object array, each (n_units, n_units) float32 (symmetrized, diag=0)
    - h_list             : object array, each (n_units,) float32
    - win_bounds         : int32 [n_windows, 2] (start_bin, end_bin)
    - bin_ms             : float
    - logloss_ind        : float32 [n_windows]  -- independent baseline (holdout)
    - logloss_fit        : float32 [n_windows]  -- fitted pseudo-log-loss (holdout)
    - logloss_holdout    : float32 [n_windows]  -- alias of logloss_fit for clarity
    - moment_m_relerr    : float32 [n_windows]  -- ||m_sim - m_obs|| / ||m_obs||
    - moment_C_relerr    : float32 [n_windows]  -- ||C_sim - C_obs||_F / ||C_obs||_F
    - settings           : JSON string (penalty, C_used, l1_ratio, cv_folds, subsample, holdout_frac, ...)

Notes:
  - Add blocked tail *holdout* per window (default 20%) and report pseudo-LL on holdout.
  - Add fast Gibbs sampler to check moment reproduction after symmetrization.
  - Record seeds/timestamps; keep dropped-windows bookkeeping hook.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

np.random.seed(0)  # global reproducibility


def _blocked_kfold_T(T: int, n_splits: int):
    """Yield (train_idx, val_idx) splits along the time axis for blocked CV."""
    fold_sizes = np.full(n_splits, T // n_splits, dtype=int)
    fold_sizes[: T % n_splits] += 1
    idx = np.arange(T)
    current = 0
    for fs in fold_sizes:
        start, stop = current, current + fs
        val_idx = idx[start:stop]
        train_idx = np.concatenate([idx[:start], idx[stop:]])
        current = stop
        yield train_idx, val_idx


def _fit_pl_window(S: np.ndarray, penalty: str, C: float, l1_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit pseudolikelihood for one window.
    S: (n_units, T) spins in {-1,+1}
    Returns (J, h).
    """
    n, T = S.shape
    Xfull = S.T  # (T, n)
    J = np.zeros((n, n), dtype=np.float32)
    h = np.zeros(n, dtype=np.float32)

    # scikit-learn setup
    if penalty not in {"l2", "elasticnet"}:
        raise ValueError("penalty must be 'l2' or 'elasticnet'")
    solver = "lbfgs" if penalty == "l2" else "saga"

    for i in range(n):
        y = (S[i] == 1).astype(int)  # {0,1}
        X = np.delete(Xfull, i, axis=1)  # drop self

        clf = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            l1_ratio=(l1_ratio if penalty == "elasticnet" else None),
            max_iter=1000,
            fit_intercept=True,
            n_jobs=None,
            verbose=0,
            tol=1e-6,
            random_state=0,
        )
        clf.fit(X, y)

        beta = clf.coef_.ravel()
        intercept = float(clf.intercept_[0])

        # Map logistic params to Ising (divide by 2); insert missing self column
        h[i] = intercept / 2.0
        row = np.zeros(n, dtype=np.float32)
        row[:i] = beta[:i] / 2.0
        row[i + 1 :] = beta[i:] / 2.0
        J[i] = row

    # Symmetrize and zero diagonal
    J = 0.5 * (J + J.T)
    np.fill_diagonal(J, 0.0)
    return J, h


def _pseudo_logloss(S: np.ndarray, J: np.ndarray, H: np.ndarray) -> float:
    """Mean binary log loss under conditional model σ(2h + 2 Σ_j J_ij s_j)."""
    logits = (2.0 * H)[None, :] + 2.0 * (S.T @ J.T)  # (T, n)
    P = 1.0 / (1.0 + np.exp(-logits))
    Y = (S == 1).astype(int).T  # (T, n)
    return float(log_loss(Y.ravel(), P.ravel(), labels=[0, 1]))


def _cv_select_C(S: np.ndarray, penalty: str, l1_ratio: float, C_grid: np.ndarray, n_folds: int) -> float:
    """Select C by blocked K-fold along time using pseudo-log-loss."""
    T = S.shape[1]
    best_C, best_loss = None, np.inf
    for C in C_grid:
        fold_losses = []
        for tr, va in _blocked_kfold_T(T, n_folds):
            J, h = _fit_pl_window(S[:, tr], penalty=penalty, C=float(C), l1_ratio=l1_ratio)
            loss = _pseudo_logloss(S[:, va], J, h)
            fold_losses.append(loss)
        mean_loss = float(np.mean(fold_losses))
        if mean_loss < best_loss:
            best_loss, best_C = mean_loss, float(C)
    return best_C


def _gibbs_sample(J: np.ndarray, h: np.ndarray, T: int, burn: int = 200, thin: int = 1, seed: int = 0) -> np.ndarray:
    """Fast Gibbs sampler returning spins (n, T)."""
    rng = np.random.default_rng(seed)
    n = J.shape[0]
    s = rng.choice([-1, 1], size=n)
    samples = []
    total = burn + T * thin
    for t in range(total):
        for i in range(n):
            field = 2.0 * h[i] + 2.0 * np.dot(J[i], s)
            p = 1.0 / (1.0 + np.exp(-field))
            s[i] = 1 if rng.random() < p else -1
        if t >= burn and ((t - burn) % thin == 0):
            samples.append(s.copy())
    return np.array(samples, dtype=np.int8).T  # (n, T)


def main():
    ap = argparse.ArgumentParser(description="Fit per-window Ising via pseudolikelihood (L2 / elastic-net).")
    ap.add_argument("--counts", required=True, help="Path to counts .npz (from 02_make_windows.py)")
    ap.add_argument("--windows", required=True, help="Path to windows .tsv (use the *filtered* TSV for acceptance)")
    ap.add_argument("--out-root", required=True, help="Directory to write outputs")
    ap.add_argument("--penalty", choices=["l2", "elasticnet"], default="l2", help="Regularization type")
    ap.add_argument("--C", type=float, default=50.0, help="Inverse regularization strength (1/l2).")
    ap.add_argument("--l1-ratio", type=float, default=0.05, help="Elastic-net mixing (only if penalty=elasticnet)")
    ap.add_argument("--cv", type=int, default=0, help="Number of CV folds along time (0 disables).")
    ap.add_argument("--subsample", type=int, default=None, help="Optional max time bins per window for speed.")
    ap.add_argument("--holdout-frac", type=float, default=0.2,
                    help="Fraction of time bins at END of each window reserved for holdout PLL.")
    ap.add_argument("--windows-filtered", action="store_true",
                    help="Set if the windows TSV is already stationarity-filtered.")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load counts and windows
    npz = np.load(args.counts, allow_pickle=False)
    X = npz["X"]              # (n_units, n_bins)
    bin_ms = float(npz["bin_ms"])
    lines = Path(args.windows).read_text().strip().splitlines()[1:]
    win_bounds = np.array([[int(c) for c in ln.split("\t")[1:3]] for ln in lines], dtype=np.int32)

    # Prepare spins {-1,+1}
    Sfull = (X > 0).astype(np.int8)
    Sfull = (2 * Sfull - 1).astype(np.int8)

    # Optional CV grid (geometric) on up to 3 windows
    C_used = args.C
    if args.cv and args.cv > 1:
        idx_sample = list(range(min(len(win_bounds), 3)))
        C_grid = np.geomspace(5.0, 200.0, num=6)  # covers l2 ~ [0.2, 0.005]
        Cs = []
        for k in idx_sample:
            s, e = win_bounds[k]
            S = Sfull[:, s:e]
            if args.subsample and S.shape[1] > args.subsample:
                idx = np.linspace(0, S.shape[1] - 1, args.subsample).astype(int)
                S = S[:, idx]
            best_C = _cv_select_C(S, penalty=args.penalty, l1_ratio=args.l1_ratio, C_grid=C_grid, n_folds=args.cv)
            Cs.append(best_C)
        C_used = float(np.median(Cs))

    # Fit all windows
    J_list, h_list = [], []
    ll_ind, ll_fit = [], []
    m_errors, C_errors = [], []
    dropped = []  # hook for bookkeeping if needed

    for w, (s, e) in enumerate(win_bounds):
        S = Sfull[:, s:e]
        if args.subsample and S.shape[1] > args.subsample:
            idx = np.linspace(0, S.shape[1] - 1, args.subsample).astype(int)
            S = S[:, idx]

        # Blocked holdout: last K% bins
        T = S.shape[1]
        K = int(max(1, round(args.holdout_frac * T)))
        tr_slice = slice(0, T - K)
        ho_slice = slice(T - K, T)
        Str, Sho = S[:, tr_slice], S[:, ho_slice]

        # Independent baseline pseudo-LL on holdout (use train frequencies)
        Ytr = (Str == 1).astype(int).T
        Yoh = (Sho == 1).astype(int).T
        p_ind_ho = Ytr.mean(axis=0, keepdims=True).repeat(Yoh.shape[0], axis=0)
        ll_ind.append(float(log_loss(Yoh.ravel(), p_ind_ho.ravel(), labels=[0, 1])))

        # PL fit on training portion
        J, h = _fit_pl_window(Str, penalty=args.penalty, C=C_used, l1_ratio=args.l1_ratio)
        J_list.append(J.astype(np.float32))
        h_list.append(h.astype(np.float32))

        # Holdout pseudo-LL under fitted conditionals
        ll_fit.append(_pseudo_logloss(Sho, J, h))

        # Moment reproduction (training portion)
        m_obs = Str.mean(axis=1)
        C_obs = (Str @ Str.T) / Str.shape[1] - np.outer(m_obs, m_obs)
        T_sim = int(min(10000, max(1000, 5 * Str.shape[1])))
        S_sim = _gibbs_sample(J, h, T_sim, seed=123 + w)
        m_sim = S_sim.mean(axis=1)
        C_sim = (S_sim @ S_sim.T) / S_sim.shape[1] - np.outer(m_sim, m_sim)
        m_err = float(np.linalg.norm(m_sim - m_obs, ord=2) / max(1e-9, np.linalg.norm(m_obs, ord=2)))
        C_err = float(np.linalg.norm(C_sim - C_obs, ord="fro") / max(1e-9, np.linalg.norm(C_obs, ord="fro")))
        m_errors.append(m_err)
        C_errors.append(C_err)

        print(f"[win {w:04d}] T={T} tr={T-K} ho={K}  pll_ind={ll_ind[-1]:.4f}  pll_fit={ll_fit[-1]:.4f}")

    settings = {
        "penalty": args.penalty,
        "C_used": C_used,
        "l1_ratio": (args.l1_ratio if args.penalty == "elasticnet" else 0.0),
        "cv_folds": int(args.cv),
        "subsample": (int(args.subsample) if args.subsample else None),
        "bin_ms": bin_ms,
        "windows_file": Path(args.windows).as_posix(),
        "counts_file": Path(args.counts).as_posix(),
        "holdout_frac": float(args.holdout_frac),
        "windows_filtered": bool(args.windows_filtered),
        "dropped_windows": dropped,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Save results (object arrays for per-window matrices)
    out_path = out_root / "ising.npz"
    np.savez_compressed(
        out_path,
        J_list=np.array(J_list, dtype=object),
        h_list=np.array(h_list, dtype=object),
        win_bounds=win_bounds,
        bin_ms=bin_ms,
        logloss_ind=np.asarray(ll_ind, dtype=np.float32),
        logloss_fit=np.asarray(ll_fit, dtype=np.float32),
        logloss_holdout=np.asarray(ll_fit, dtype=np.float32),  # explicit alias
        moment_m_relerr=np.asarray(m_errors, dtype=np.float32),
        moment_C_relerr=np.asarray(C_errors, dtype=np.float32),
        settings=json.dumps(settings),
    )
    print(f"[ok] Ising per-window -> {out_path}")


if __name__ == "__main__":
    main()
