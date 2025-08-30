#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
*untested
!!!Update comments and header!!!

Select γ* per window with:
  - HO-closure: median ΔΩ(γ) ≤ 5th percentile of shuffled ΔΩ (real vs Ising)
  - PL-sanity: held-out log-loss(γ) ≤ (1+1%) * log-loss(γ=0)
Also excludes inadmissible γ (clip_rate>5% or surrogate QC fail).

Notes:
- Compute p05 using the Ising surrogate path saved in step-05 NPZ
- Enforce monotonic log-loss fallback: if no closure and log-loss is not monotonically improving with γ>0 ⇒ γ*=0
"""

import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy import special as sps
from toolbox.gcmi import copnorm
from toolbox.oinfo import o_information_boot

def copnorm_midrank(X):
    R = np.vstack([rankdata(row, method="average") for row in X])
    U = R / (R.shape[1] + 1.0)
    return sps.ndtri(U)

def shuffled_p05(X_real, Xi, tri, H_bins, estimator, rng):
    Xr = copnorm_midrank(X_real)
    Xi = copnorm_midrank(Xi)
    n, T = Xr.shape
    vals = []
    ind_all = np.arange(T, dtype=np.int32)
    for _ in range(50):
        Xp = np.empty_like(Xr)
        shifts = rng.integers(low=H_bins, high=T, size=n, endpoint=False)
        for i in range(n):
            Xp[i] = np.roll(Xr[i], int(shifts[i]) % T)
        for (a,b,c) in tri:
            indv = np.array([a,b,c], dtype=np.int32)
            try:
                Om_p = float(o_information_boot(Xp, ind_all, indv, estimator))
                Om_i = float(o_information_boot(Xi, ind_all, indv, estimator))
            except np.linalg.LinAlgError:
                eps = 1e-6
                Om_p = float(o_information_boot(Xp + eps*np.random.default_rng(0).normal(size=Xp.shape), ind_all, indv, estimator))
                Om_i = float(o_information_boot(Xi + eps*np.random.default_rng(1).normal(size=Xi.shape), ind_all, indv, estimator))
            vals.append(Om_p - Om_i)
    return float(np.percentile(vals, 5.0)) if len(vals) else np.nan

def is_monotonic_improving(gammas, losses):
    # consider only γ>0, check nonincreasing losses
    arr = [losses[i] for i,g in enumerate(gammas) if g > 0]
    if len(arr) < 2: return False
    diffs = np.diff(arr)
    return bool(np.all(diffs <= 1e-12))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curved", required=True)                   # curved_fit.npz
    ap.add_argument("--omega-real", required=True)               # omega_equal_time.npz (from real vs Ising)
    ap.add_argument("--counts", required=True)                   # counts_bin10ms.npz
    ap.add_argument("--surrogates", required=True)               # root with gamma_*/omega_equal_time.npz
    ap.add_argument("--candidates", required=True)               # triangles_per_window
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--horizon-ms", type=float, default=500.0)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    wdf = pd.read_csv(args.windows, sep="\t")
    tri_list = list(np.load(args.candidates, allow_pickle=True)["triangles_per_window"])

    npz_curv = np.load(args.curved, allow_pickle=True)
    gammas = list(npz_curv["gammas"])
    ll_grid = list(npz_curv["cv_logloss"])  # per γ, per window
    clip_grid = list(npz_curv["clip_rate"])

    # Read curved surrogate QC flags
    qc_path = Path(args.surrogates) / "curved_surrogate_qc.tsv"
    if not qc_path.exists():
        raise FileNotFoundError(f"Missing QC TSV: {qc_path}")
    qc_df = pd.read_csv(qc_path, sep="\t")

    # Load ΔΩ(γ) from each gamma_* dir
    Delta_gamma = []
    for g in gammas:
        tag = f"gamma_{g:+.2f}".replace("+","p").replace("-","m")
        om_g_path = Path(args.surrogates) / tag / "omega_equal_time.npz"
        if not om_g_path.exists():
            raise FileNotFoundError(f"Missing Ω for {tag}: {om_g_path}")
        npz = np.load(om_g_path, allow_pickle=True)
        Delta_gamma.append(list(npz["DeltaOmega_iso"]))

    # Load counts and Ising surrogate path to compute p05 baseline
    npz_cnt = np.load(args.counts, allow_pickle=True)
    ckey = "X" if "X" in npz_cnt.files else ("counts" if "counts" in npz_cnt.files else None)
    if ckey is None: raise ValueError("--counts npz missing 'X' or 'counts'")
    Xfull = npz_cnt[ckey].astype(np.float64)
    bin_ms = float(npz_cnt["bin_ms"]) if "bin_ms" in npz_cnt.files else 10.0
    H_bins = max(1, int(round(args.horizon_ms / bin_ms)))

    npz_om = np.load(args.omega_real, allow_pickle=True)
    if "surrogate_path" not in npz_om.files:
        raise ValueError("omega_real npz must contain 'surrogate_path' to compute shuffled baseline.")
    iso_path = npz_om["surrogate_path"].item()
    npz_iso = np.load(iso_path, allow_pickle=True)
    Iso_list = list(npz_iso["counts_iso"])

    rng = np.random.default_rng(args.seed)

    # Compute per-window p05 baseline
    p05 = []
    for w, row in wdf.iterrows():
        s, e = int(row["start_bin"]), int(row["end_bin"])
        Xr = Xfull[:, s:e]
        Xi = np.asarray(Iso_list[w], dtype=np.float64)
        tri = np.asarray(tri_list[w], dtype=np.int32)
        p05.append(shuffled_p05(Xr, Xi, tri, H_bins, "gcmi", rng))

    # Selection
    results = {}
    records = []
    for w, row in wdf.iterrows():
        # admissible γ: clip_rate <= 0.05 AND surrogate QC accepted==1
        admissible = []
        for g_idx, g in enumerate(gammas):
            clip_ok = (float(clip_grid[g_idx][w]) <= 0.05)
            qc_ok = int(qc_df[(qc_df.win_id == int(row["win_id"])) & (np.isclose(qc_df.gamma, g))]["accepted"].iloc[0]) == 1
            admissible.append(bool(clip_ok and qc_ok))

        Dmed = []
        ll_curve = []
        for g_idx, g in enumerate(gammas):
            D = np.asarray(Delta_gamma[g_idx][w], dtype=np.float64)
            Dmed.append(float(np.nanmedian(D)) if D.size else np.nan)
            ll_curve.append(float(ll_grid[g_idx][w]))

        # PL-sanity relative to γ=0 (if present)
        try:
            idx0 = gammas.index(0.0)
            ll0 = ll_curve[idx0]
        except ValueError:
            ll0 = ll_curve[0]  # fallback
        pl_ok = [(ll <= 1.01 * ll0) for ll in ll_curve]

        # HO-closure vs baseline p05
        base = p05[w]
        ho_ok = [ (Dmed[g_idx] <= base) if admissible[g_idx] else False for g_idx in range(len(gammas)) ]

        # pick smallest γ>0 with ho_ok & pl_ok & admissible
        gamma_star = 0.0
        closed = False
        for g_idx, g in enumerate(gammas):
            if g <= 0.0: continue
            if admissible[g_idx] and ho_ok[g_idx] and pl_ok[g_idx]:
                gamma_star = g; closed = True; break

        # fallback if no closure
        no_closure = False
        if not closed:
            if is_monotonic_improving(gammas, ll_curve):
                # pick argmin log-loss among admissible γ>0
                cand = [(ll_curve[i], gammas[i]) for i in range(len(gammas)) if gammas[i] > 0 and admissible[i]]
                if len(cand) > 0:
                    gamma_star = float(min(cand, key=lambda z: z[0])[1])
                else:
                    gamma_star = 0.0
            else:
                gamma_star = 0.0
                no_closure = True

        results[int(row["win_id"])] = dict(
            gamma_star=float(gamma_star),
            ho_closure=bool(closed),
            no_closure=bool(no_closure),
            pl_sanity_at_star=bool(pl_ok[gammas.index(gamma_star)] if gamma_star in gammas else True),
            med_Delta_curve=[float(x) for x in Dmed],
            logloss_curve=[float(x) for x in ll_curve],
            admissible=admissible,
            p05=float(base),
            gammas=[float(x) for x in gammas]
        )
        records.append(dict(win_id=int(row["win_id"]),
                            gamma_star=float(gamma_star),
                            ho_closure=int(closed),
                            no_closure=int(no_closure),
                            p05=float(base),
                            ll0=float(ll0)))

        print(f"[win {w:04d}] γ*={gamma_star:+.2f}  closure={closed}  fallback_no_closure={no_closure}")

    with open(out_root / "gamma_star.json", "w") as f:
        json.dump(dict(per_window=results), f, indent=2)
    pd.DataFrame(records).to_csv(out_root / "gamma_star.tsv", sep="\t", index=False)
    pd.DataFrame(dict(win_id=wdf["win_id"], p05=p05)).to_csv(out_root / "baseline_percentiles.tsv", sep="\t", index=False)
    print(f"[ok] wrote gamma_star.json / gamma_star.tsv / baseline_percentiles.tsv")

if __name__ == "__main__":
    main()
