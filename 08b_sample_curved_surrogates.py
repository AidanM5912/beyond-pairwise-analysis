#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
*untested
!!!Update comments and header!!!

Sample γ-surrogates per window using Gibbs with deformed conditionals.


Writes per-γ subdirs with counts_curved.npz and a consolidated QC TSV.
"""

import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd

def sigma_gamma(x, gamma):
    if abs(gamma) < 1e-12:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))
    one_p = np.maximum(1.0 + gamma * x, 1e-6)
    one_m = np.maximum(1.0 - gamma * x, 1e-6)
    a = np.power(one_p, 1.0 / gamma)
    b = np.power(one_m, 1.0 / gamma)
    return a / (a + b + 1e-24)

def gibbs_deformed(J, h, T, gamma, burn_sweeps, thin, seed):
    rng = np.random.default_rng(seed)
    n = J.shape[0]
    s = rng.choice([-1, 1], size=n).astype(np.int8)
    samples = []
    total_sweeps = burn_sweeps + T * thin
    xmax = (1.0 / abs(gamma) - 1e-6) if abs(gamma) > 1e-12 else 50.0
    for t in range(total_sweeps):
        for i in range(n):
            field = 2.0 * (h[i] + np.dot(J[i], s))
            field = np.clip(field, -xmax, xmax)
            p = sigma_gamma(field, gamma)
            s[i] = 1 if rng.random() < p else -1
        if t >= burn_sweeps and ((t - burn_sweeps) % thin == 0):
            samples.append(s.copy())
    S = np.array(samples, dtype=np.int8).T
    return ((S + 1) // 2).astype(np.uint8)

def window_moment_errors(X_real01, X_sim01):
    Sr = (X_real01 * 2.0 - 1.0).astype(np.float64)
    Ss = (X_sim01  * 2.0 - 1.0).astype(np.float64)
    mr = Sr.mean(axis=1); ms = Ss.mean(axis=1)
    Cr = (Sr @ Sr.T) / Sr.shape[1] - np.outer(mr, mr)
    Cs = (Ss @ Ss.T) / Ss.shape[1] - np.outer(ms, ms)
    m_err = float(np.max(np.abs(ms - mr)))
    denom = np.linalg.norm(Cr, ord='fro') + 1e-12
    C_err = float(np.linalg.norm(Cs - Cr, ord='fro') / denom)
    return m_err, C_err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curved", required=True)
    ap.add_argument("--counts", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--burn-mult", type=float, default=5.0)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--m-tol", type=float, default=0.02)
    ap.add_argument("--C-rel-tol", type=float, default=0.10)
    args = ap.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    npz_curv = np.load(args.curved, allow_pickle=True)
    gammas = list(npz_curv["gammas"])
    J_grid  = list(npz_curv["J_grid"])
    h_grid  = list(npz_curv["h_grid"])
    win_bounds = np.array(npz_curv["win_bounds"])
    wdf = pd.read_csv(args.windows, sep="\t")

    npz_cnt = np.load(args.counts, allow_pickle=True)
    ckey = "X" if "X" in npz_cnt.files else ("counts" if "counts" in npz_cnt.files else None)
    if ckey is None:
        raise ValueError(f"{args.counts} missing 'X' or 'counts'")
    Xfull = npz_cnt[ckey].astype(np.uint8)
    bin_ms = float(npz_cnt["bin_ms"]) if "bin_ms" in npz_cnt.files else 10.0

    qc_rows = []
    for g_idx, gamma in enumerate(gammas):
        subdir = out_root / f"gamma_{gamma:+.2f}".replace("+","p").replace("-","m")
        subdir.mkdir(parents=True, exist_ok=True)
        counts_list, seeds, m_errs, C_errs, flags = [], [], [], [], []
        for w, row in wdf.iterrows():
            s, e = int(row["start_bin"]), int(row["end_bin"])
            if win_bounds.shape[0] == len(wdf):
                if not (int(win_bounds[w,0]) == s and int(win_bounds[w,1]) == e):
                    raise ValueError(f"Window {w}: curved win_bounds mismatch TSV")
            T = e - s
            Xr = Xfull[:, s:e]
            J = J_grid[g_idx][w].astype(np.float64)
            h = h_grid[g_idx][w].astype(np.float64)
            burn = int(round(args.burn_mult * J.shape[0]))
            tried = 0; ok = False
            while tried < 3 and not ok:
                seed = int(args.seed + 1000*g_idx + w + tried)
                Xsim = gibbs_deformed(J, h, T=T, gamma=gamma, burn_sweeps=burn, thin=args.thin, seed=seed)
                m_err, C_err = window_moment_errors(Xr, Xsim)
                ok = (m_err <= args.m_tol) and (C_err <= args.C_rel_tol)
                if not ok: burn = int(round(burn * 2.0))
                tried += 1
            counts_list.append(Xsim); seeds.append(seed)
            m_errs.append(m_err); C_errs.append(C_err); flags.append(int(ok))
            qc_rows.append(dict(win_id=int(row["win_id"]), gamma=float(gamma),
                                m_err=m_err, C_rel_err=C_err, accepted=int(ok)))
            print(f"[γ {gamma:+.2f} win {w:04d}] T={T}  m_err={m_err:.4f}  C_rel_err={C_err:.4f}  accepted={ok}")

        np.savez_compressed(
            subdir / "counts_curved.npz",
            counts_curved=np.array(counts_list, dtype=object),
            seeds=np.array(seeds, dtype=np.int64),
            win_bounds=win_bounds if win_bounds.shape[0]==len(wdf) else np.zeros((0,2), dtype=int),
            windows_tsv=args.windows,
            counts_path=args.counts,
            curved_path=args.curved,
            settings=json.dumps(dict(
                burn_mult=args.burn_mult, thin=args.thin, seed=args.seed,
                m_tol=args.m_tol, C_rel_tol=args.C_rel_tol,
                bin_ms=float(bin_ms), timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            )),
        )

    pd.DataFrame(qc_rows).to_csv(out_root / "curved_surrogate_qc.tsv", sep="\t", index=False)
    print(f"[ok] wrote γ-surrogates under {out_root}")

if __name__ == "__main__":
    main()
