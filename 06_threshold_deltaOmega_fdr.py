#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
*untested

Permutation-FDR thresholding of ΔΩ per window (one-sided, ΔΩ>0).

Notes:
- Per-triplet permutation nulls (no pooled null)
- Case-correct import (toolbox.oinfo)
- Robust counts/surrogate key handling is inherited from step-05; here we only need Ω outputs + raw counts for permutations.

Inputs
  --omega       omega_equal_time.npz (from step-05; contains ΔΩ observed)
  --counts      counts_bin10ms.npz (for real data)
  --surrogate   counts_iso.npz (Ising) — used to compute Ω_iso in the null
  --windows     windows_*_ov50.filtered.tsv
  --candidates  candidates.npz
  --out-root
  --horizon-ms  minimum circular shift (default 500)
  --perm        permutation reps (default 50)
  --estimator   gcmi|lin_est (default gcmi)
  --seed

Outputs
  - fdr_thresholds.tsv (per window: q, tau, discoveries)
  - S_HO.npz (boolean masks per window for ΔΩ>tau)
"""

import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from toolbox.gcmi import copnorm
from toolbox.oinfo import o_information_boot

def bh_fdr(pvals, q):
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    thresh = q * (np.arange(1, m+1) / m)
    passed = p[order] <= thresh
    k = np.where(passed)[0].max()+1 if np.any(passed) else 0
    if k == 0:
        return None, np.zeros(m, dtype=bool)
    cutoff = p[order][k-1]
    mask = p <= cutoff
    return cutoff, mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omega", required=True)
    ap.add_argument("--counts", required=True)
    ap.add_argument("--surrogate", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--horizon-ms", type=float, default=500.0)
    ap.add_argument("--perm", type=int, default=50)
    ap.add_argument("--q", type=float, default=0.1, help="FDR level (e.g., 0.05, 0.10, 0.20)")
    ap.add_argument("--estimator", choices=["gcmi","lin_est"], default="gcmi")
    ap.add_argument("--seed", type=int, default=999)
    args = ap.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    wdf = pd.read_csv(args.windows, sep="\t")
    tri_list = list(np.load(args.candidates, allow_pickle=True)["triangles_per_window"])

    npz_cnt = np.load(args.counts, allow_pickle=True)
    ckey = "X" if "X" in npz_cnt.files else ("counts" if "counts" in npz_cnt.files else None)
    if ckey is None:
        raise ValueError("--counts npz missing 'X' or 'counts'")
    Xfull = npz_cnt[ckey].astype(np.float64)
    bin_ms = float(npz_cnt["bin_ms"]) if "bin_ms" in npz_cnt.files else 10.0
    H_bins = max(1, int(round(args.horizon_ms / bin_ms)))

    npz_iso = np.load(args.surrogate, allow_pickle=True)
    skey = "counts_iso" if "counts_iso" in npz_iso.files else None
    if skey is None:
        raise ValueError("--surrogate must be an Ising counts_iso.npz for nulls")
    Iso_list = list(npz_iso[skey])

    npz_om = np.load(args.omega, allow_pickle=True)
    Delta = list(npz_om["DeltaOmega_iso"])

    rows = []
    masks = []

    for w, row in enumerate(wdf.itertuples(index=False)):
        s, e = int(row.start_bin), int(row.end_bin)
        Xr = copnorm(Xfull[:, s:e])
        Xi = copnorm(np.asarray(Iso_list[w], dtype=np.float64))
        T = Xr.shape[1]
        ind_all = np.arange(T, dtype=np.int32)
        tri = np.asarray(tri_list[w], dtype=np.int32)

        # observed ΔΩ
        D = np.asarray(Delta[w], dtype=np.float64)
        M = len(tri)
        per_trip_null = np.zeros((M, args.perm), dtype=np.float32)

        # build permutations once per p and compute triplet-specific ΔΩ
        for p in range(args.perm):
            Xp = np.empty_like(Xr)
            shifts = rng.integers(low=H_bins, high=T, size=Xr.shape[0], endpoint=False)
            for i in range(Xr.shape[0]):
                Xp[i] = np.roll(Xr[i], int(shifts[i]) % T)
            for m, (i,j,k) in enumerate(tri):
                indv = np.array([i,j,k], dtype=np.int32)
                try:
                    Om_p = float(o_information_boot(Xp, ind_all, indv, args.estimator))
                    Om_i = float(o_information_boot(Xi, ind_all, indv, args.estimator))
                except np.linalg.LinAlgError:
                    eps = 1e-6
                    Om_p = float(o_information_boot(Xp + eps*np.random.default_rng(0).normal(size=Xp.shape), ind_all, indv, args.estimator))
                    Om_i = float(o_information_boot(Xi + eps*np.random.default_rng(1).normal(size=Xi.shape), ind_all, indv, args.estimator))
                per_trip_null[m, p] = Om_p - Om_i

        # per-triplet p-values (upper-tail, ΔΩ > 0)
        pvals = (np.sum(per_trip_null >= D[:, None], axis=1) + 1.0) / (args.perm + 1.0)

        # BH-FDR across triplets
        cutoff_p, mask = bh_fdr(pvals, q=args.q)
        masks.append(mask)

        # define τ as ΔΩ value at last accepted rank (if any)
        if np.any(mask):
            sort_idx = np.argsort(-D)  # descending ΔΩ
            accepted = sort_idx[:np.sum(mask)]
            tau = float(np.min(D[accepted]))
            n_disc = int(np.sum(mask))
        else:
            tau = float("nan")
            n_disc = 0

        rows.append(dict(win_id=int(getattr(row, "win_id")), q=args.q, tau=tau,
                         discoveries=n_disc, perm=args.perm, horizon_ms=args.horizon_ms))

        print(f"[win {w:04d}] q={args.q:.2f}  tau={tau:.4f}  discoveries={n_disc}")

    np.savez_compressed(out_root / "S_HO.npz", masks=np.array(masks, dtype=object))
    pd.DataFrame(rows).to_csv(out_root / "fdr_thresholds.tsv", sep="\t", index=False)
    print(f"[ok] wrote S_HO.npz and fdr_thresholds.tsv")

if __name__ == "__main__":
    main()
