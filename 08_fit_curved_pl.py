#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
*untested

!!!Update comments and header!!!

Fit curved pseudolikelihood (Tsallis / deformed logistic) per window over a γ grid.

Notes:
- TV smoothness watchdog compares J(γ_k) vs J(γ_{k-1}) **for the same window**.

"""

import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---------- deformed logistic utilities ----------

def exp_gamma(x, gamma, eps=1e-6):
    if abs(gamma) < 1e-12:
        return np.exp(x)
    one = 1.0 + gamma * x
    one = np.maximum(one, eps)
    return np.power(one, 1.0 / gamma)

def log_exp_gamma(x, gamma, eps=1e-6):
    if abs(gamma) < 1e-12:
        return x
    one = 1.0 + gamma * x
    one = np.maximum(one, eps)
    return (1.0 / gamma) * np.log(one)

def sigma_gamma(x, gamma, eps=1e-6):
    if abs(gamma) < 1e-12:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))
    la = log_exp_gamma(x, gamma, eps)
    lb = log_exp_gamma(-x, gamma, eps)
    lden = np.logaddexp(la, lb)
    return np.exp(la - lden)

def dsig_dx(x, gamma, eps=1e-6):
    if abs(gamma) < 1e-12:
        p = 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))
        return p * (1.0 - p)
    a = exp_gamma(x, gamma, eps)
    b = exp_gamma(-x, gamma, eps)
    denom1 = (1.0 - (gamma * x)**2)
    denom1 = np.maximum(denom1, eps)
    denom2 = (a + b)**2 + 1e-24
    return (2.0 * a * b) / (denom1 * denom2)

def node_loss_grad(theta, S_pm, i, id_tr, gamma, lamJ, lamH, clip_eps):
    n, T = S_pm.shape
    y = ((S_pm[i, id_tr] + 1) // 2).astype(np.float64)
    X = np.delete(S_pm[:, id_tr], i, axis=0)
    h = theta[0]
    J = theta[1:]
    x = 2.0 * (h + np.einsum("k,kt->t", J, X))
    if abs(gamma) > 1e-12:
        xmax = (1.0 / abs(gamma)) - clip_eps
        x_clip = np.clip(x, -xmax, xmax)
    else:
        x_clip = np.clip(x, -50.0, 50.0)
    p = sigma_gamma(x_clip, gamma)
    ll = - (y * np.log(p + 1e-24) + (1.0 - y) * np.log(1.0 - p + 1e-24)).sum()
    reg = lamJ * np.dot(J, J) + lamH * (h * h)
    loss = ll + reg
    if abs(gamma) < 1e-12:
        dldx = (p - y)
    else:
        dpdx = dsig_dx(x_clip, gamma)
        dldx = ((p - y) / (p * (1.0 - p) + 1e-24)) * dpdx
    grad_h = (dldx * 2.0).sum() + 2.0 * lamH * h
    grad_J = np.einsum("t,kt->k", dldx * 2.0, X) + 2.0 * lamJ * J
    clip_rate = 0.0
    if abs(gamma) > 1e-12:
        xmax = (1.0 / abs(gamma)) - clip_eps
        clip_rate = float(np.mean((x <= -xmax) | (x >= xmax)))
    return loss, np.concatenate(([grad_h], grad_J)), clip_rate

def contiguous_kfold(T, K):
    bounds = np.linspace(0, T, K + 1, dtype=int)
    return [(np.setdiff1d(np.arange(T), np.arange(bounds[k], bounds[k+1]), assume_unique=False),
             np.arange(bounds[k], bounds[k+1])) for k in range(K)]

def heldout_scores(S_pm, i, h, Jrow, id_val, gamma):
    y = ((S_pm[i, id_val] + 1) // 2).astype(np.float64)
    X = np.delete(S_pm[:, id_val], i, axis=0)
    x = 2.0 * (h + np.einsum("k,kt->t", Jrow, X))
    if abs(gamma) > 1e-12:
        xmax = (1.0 / abs(gamma)) - 1e-6
        x = np.clip(x, -xmax, xmax)
    else:
        x = np.clip(x, -50.0, 50.0)
    p = sigma_gamma(x, gamma)
    logloss = float(-np.mean(y * np.log(p + 1e-24) + (1.0 - y) * np.log(1.0 - p + 1e-24)))
    brier = float(np.mean((y - p) ** 2))
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    ece = 0.0
    for b in range(10):
        mask = (idx == b)
        if not np.any(mask): continue
        pb = p[mask].mean(); yb = y[mask].mean()
        ece += (np.sum(mask) / len(p)) * abs(pb - yb)
    return logloss, brier, ece

def fit_window_gamma(S_pm, gamma, lamJ, lamH, cv_folds, max_iter, init_hJ):
    n, T = S_pm.shape
    folds = contiguous_kfold(T, cv_folds)
    h_full = np.zeros(n, dtype=np.float64)
    J_full = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        h0 = init_hJ["h"][i]
        Ji0 = np.delete(init_hJ["J"][i], i)
        theta0 = np.concatenate(([h0], Ji0))
        def f(th):
            return node_loss_grad(th, S_pm, i, np.arange(T), gamma, lamJ, lamH, clip_eps=1e-6)[:2]
        res = minimize(lambda th: f(th)[0], theta0, jac=lambda th: f(th)[1],
                       method="L-BFGS-B", options=dict(maxiter=max_iter, ftol=1e-9))
        th = res.x
        h_full[i] = th[0]
        row = np.zeros(n, dtype=np.float64)
        Ji = th[1:]
        row[:i] = Ji[:i]; row[i+1:] = Ji[i:]
        J_full[i, :] = row
    J_sym = 0.5 * (J_full + J_full.T); np.fill_diagonal(J_sym, 0.0)
    h_sym = h_full.copy()

    # CV metrics
    ll_list, br_list, ece_list = [], [], []
    for (id_tr, id_val) in folds:
        h_cv = np.zeros(n, dtype=np.float64)
        J_cv = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            h0 = h_sym[i]
            Ji0 = np.delete(J_sym[i], i)
            theta0 = np.concatenate(([h0], Ji0))
            def fcv(th):
                return node_loss_grad(th, S_pm, i, id_tr, gamma, lamJ, lamH, clip_eps=1e-6)[:2]
            res = minimize(lambda th: fcv(th)[0], theta0, jac=lambda th: fcv(th)[1],
                           method="L-BFGS-B", options=dict(maxiter=max_iter, ftol=1e-9))
            th = res.x
            h_cv[i] = th[0]
            row = np.zeros(n, dtype=np.float64)
            Ji = th[1:]; row[:i] = Ji[:i]; row[i+1:] = Ji[i:]
            J_cv[i, :] = row
        J_cv = 0.5 * (J_cv + J_cv.T); np.fill_diagonal(J_cv, 0.0)
        ll_i, br_i, ece_i = [], [], []
        for i in range(n):
            ll, br, ece = heldout_scores(S_pm, i, h_cv[i], np.delete(J_cv[i], i), id_val, gamma)
            ll_i.append(ll); br_i.append(br); ece_i.append(ece)
        ll_list.append(np.mean(ll_i)); br_list.append(np.mean(br_i)); ece_list.append(np.mean(ece_i))
    return J_sym, h_sym, float(np.mean(ll_list)), float(np.mean(br_list)), float(np.mean(ece_list))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--ising", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--gammas", default="-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4")
    ap.add_argument("--lambda-J", type=float, default=1e-2)
    ap.add_argument("--lambda-h", type=float, default=1e-3)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--tv-thresh", type=float, default=0.50)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    wdf = pd.read_csv(args.windows, sep="\t")
    npz_cnt = np.load(args.counts, allow_pickle=True)
    ckey = "X" if "X" in npz_cnt.files else ("counts" if "counts" in npz_cnt.files else None)
    if ckey is None:
        raise ValueError(f"{args.counts} missing 'X' or 'counts'")
    X = npz_cnt[ckey].astype(np.float64)

    npz_iso = np.load(args.ising, allow_pickle=True)
    J0_list = list(npz_iso["J_list"]); h0_list = list(npz_iso["h_list"])
    win_bounds = np.array(npz_iso["win_bounds"]) if "win_bounds" in npz_iso.files else None

    gammas = [float(g) for g in args.gammas.split(",")]

    J_grid, h_grid = [], []
    ll_grid, br_grid, ece_grid, clip_grid, tv_viol_grid = [], [], [], [], []
    rows = []

    for g_idx, gamma in enumerate(gammas):
        J_list_g, h_list_g = [], []
        ll_list_g, br_list_g, ece_list_g, clip_list_g, tv_list_g = [], [], [], [], []

        for w, row in wdf.iterrows():
            s, e = int(row["start_bin"]), int(row["end_bin"])
            if win_bounds is not None:
                if not (int(win_bounds[w,0]) == s and int(win_bounds[w,1]) == e):
                    raise ValueError(f"Window {w}: ising bounds {tuple(win_bounds[w])} != TSV {(s,e)}")
            S_pm = (X[:, s:e] * 2.0 - 1.0).astype(np.float64)
            init = dict(J=J0_list[w].astype(np.float64), h=h0_list[w].astype(np.float64)) if g_idx == 0 \
                   else dict(J=J_grid[-1][w], h=h_grid[-1][w])

            Jw, hw, ll, br, ece = fit_window_gamma(
                S_pm, gamma, lamJ=args.__dict__["lambda-J"], lamH=args.__dict__["lambda-h"],
                cv_folds=args.cv_folds, max_iter=args.max_iter, init_hJ=init
            )

            # TV watchdog (compare across γ for the same window)
            tv_flag = False
            if g_idx > 0:
                prevJ = J_grid[-1][w]
                num = np.sum(np.abs(Jw - prevJ))
                den = np.sum(np.abs(prevJ)) + 1e-12
                tv_flag = (num / den) > args.tv_thresh

            # rough clip proxy: evaluate once on full data
            if abs(gamma) > 1e-12:
                fields = 2.0 * (hw[:, None] + Jw @ S_pm)
                xmax = (1.0 / abs(gamma)) - 1e-6
                clip_rate = float(np.mean((fields <= -xmax) | (fields >= xmax)))
            else:
                clip_rate = 0.0

            J_list_g.append(Jw); h_list_g.append(hw)
            ll_list_g.append(ll); br_list_g.append(br); ece_list_g.append(ece)
            clip_list_g.append(clip_rate); tv_list_g.append(tv_flag)
            rows.append(dict(win_id=int(wdf.loc[w, "win_id"]), gamma=gamma, logloss=ll, brier=br,
                             ece10=ece, clip_rate=clip_rate, tv_violation=int(tv_flag)))
            print(f"[γ {gamma:+.2f} win {w:04d}] ll={ll:.4f}  brier={br:.4f}  ece10={ece:.4f}  clip={clip_rate:.3f}  tvViol={tv_flag}")

        J_grid.append(J_list_g); h_grid.append(h_list_g)
        ll_grid.append(ll_list_g); br_grid.append(br_list_g); ece_grid.append(ece_list_g)
        clip_grid.append(clip_list_g); tv_viol_grid.append(tv_list_g)

    np.savez_compressed(
        out_root / "curved_fit.npz",
        gammas=np.array(gammas, dtype=np.float64),
        J_grid=np.array(J_grid, dtype=object),
        h_grid=np.array(h_grid, dtype=object),
        cv_logloss=np.array(ll_grid, dtype=object),
        cv_brier=np.array(br_grid, dtype=object),
        ece10=np.array(ece_grid, dtype=object),
        clip_rate=np.array(clip_grid, dtype=object),
        tv_violation=np.array(tv_viol_grid, dtype=object),
        windows_tsv=args.windows,
        counts_path=args.counts,
        ising_path=args.ising,
        win_bounds=win_bounds if win_bounds is not None else np.zeros((0,2), dtype=int),
        settings=json.dumps(dict(
            lambda_J=args.__dict__["lambda-J"], lambda_h=args.__dict__["lambda-h"],
            cv_folds=args.cv_folds, max_iter=args.max_iter, tv_thresh=args.tv_thresh,
            seed=args.seed, timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )),
    )
    pd.DataFrame(rows).to_csv(Path(args.out_root) / "curved_fit.summaries.tsv", sep="\t", index=False)
    print(f"[ok] wrote curved_fit.npz and curved_fit.summaries.tsv")

if __name__ == "__main__":
    main()
