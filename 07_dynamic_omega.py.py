#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
*untested
!!!Update comments and header!!!

Dynamic O-information per window on a fixed triplet set.
- Inputs: counts_bin10ms.npz, windows_*.filtered.tsv, candidates npz, labels.npz (per-window Y)
- Output: omega/domega.npz + domega_summaries.tsv

Notes:
  * Use block bootstrap WITH replacement; save mean and SE per triplet.
  * Mid-rank copnorm; drop silent units consistently with Ω stage.
"""
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy import special as sps

def copnorm_midrank(X_vars_time: np.ndarray) -> np.ndarray:
    X = np.asarray(X_vars_time)
    R = np.vstack([rankdata(row, method="average") for row in X])
    U = R / (R.shape[1] + 1.0)
    return sps.ndtri(U)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True, help="Real counts npz")
    ap.add_argument("--windows", required=True, help="windows_*_filtered.tsv")
    ap.add_argument("--candidates", required=True, help="candidates npz (triangles_per_window)")
    ap.add_argument("--labels", required=True, help="NPZ with per-window labels: Y_list (list of length n_windows, shape (T,))")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--model-order", type=int, default=10, help="m lags (m*Δt ≈ 50–150 ms recommended)")
    ap.add_argument("--bootstrap", type=int, default=100, help="bootstrap replicates (moving-block with replacement)")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--hoi-toolbox-dir", default=None)
    ap.add_argument("--estimator", choices=["gcmi","lin_est"], default="gcmi")
    args = ap.parse_args()

    if args.hoi_toolbox_dir:
        sys.path.insert(0, args.hoi_toolbox_dir)
    from toolbox.dOinfo import o_information_lagged_boot  # inputs already copnormed inside our wrapper

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    win_df = pd.read_csv(args.windows, sep="\t")
    npz_cnt = np.load(args.counts, allow_pickle=True)
    npz_cand = np.load(args.candidates, allow_pickle=True)
    npz_lab = np.load(args.labels, allow_pickle=True)

    key = "X" if "X" in npz_cnt.files else ("counts" if "counts" in npz_cnt.files else None)
    if key is None:
        raise ValueError(f"{args.counts} must contain 'X' or 'counts'")
    Xfull = npz_cnt[key]                 # (n, T_total)
    bin_ms = float(npz_cnt["bin_ms"]) if "bin_ms" in npz_cnt.files else 10.0
    tri_list = list(npz_cand["triangles_per_window"])
    Y_list  = list(npz_lab["Y_list"])  # each (T_w,)

    if len(Y_list) != len(win_df):
        raise ValueError("--labels Y_list length must match windows")

    rng = np.random.default_rng(args.seed)
    chunklength = max(1, int(round(50.0 / bin_ms)))  # 50 ms blocks

    dOmega_list = []
    dOmega_SE_list = []
    summaries = []

    for w, row in win_df.iterrows():
        s, e = int(row["start_bin"]), int(row["end_bin"])
        T = e - s
        triplets = tri_list[w]
        if triplets.size == 0:
            dOmega_list.append(np.array([], dtype=np.float32))
            dOmega_SE_list.append(np.array([], dtype=np.float32))
            summaries.append(dict(win_id=int(row["win_id"]), n_triplets=0))
            continue

        Xr0 = Xfull[:, s:e].astype(float)  # (n, T)
        Y0  = Y_list[w].astype(float)      # (T,)
        # Drop silent units and refilter triplets
        active = (Xr0.sum(axis=1) > 0)
        if not np.all(active):
            keep_trip = active[triplets].all(axis=1)
            triplets = triplets[keep_trip]
            Xr0 = Xr0[active]
            if triplets.size == 0:
                dOmega_list.append(np.array([], dtype=np.float32))
                dOmega_SE_list.append(np.array([], dtype=np.float32))
                summaries.append(dict(win_id=int(row["win_id"]), n_triplets=0))
                continue

        # Copula normalize X (time x vars) and Y
        Xr = copnorm_midrank(Xr0).T  # (T, n_active)
        Y  = copnorm_midrank(Y0.reshape(1, -1)).ravel()  # (T,)

        starts = np.arange(0, T - chunklength + 1, dtype=np.int32)

        M = len(triplets)
        dO_vals = np.zeros(M, dtype=np.float32)
        dO_bs = np.zeros((args.bootstrap, M), dtype=np.float32)

        for m, (i, j, k) in enumerate(triplets):
            indvar = np.array([i, j, k], dtype=np.int32)
            # Bootstrap with replacement over 50 ms blocks
            for b in range(args.bootstrap):
                picks = rng.choice(starts, size=max(1, len(starts)), replace=True)
                o = float(o_information_lagged_boot(Y, Xr, args.model_order, picks, chunklength, indvar, args.estimator))
                dO_bs[b, m] = o
            dO_vals[m] = float(np.mean(dO_bs[:, m]))

        dOmega_list.append(dO_vals.astype(np.float32))
        dOmega_SE_list.append(dO_bs.std(axis=0, ddof=1))
        summaries.append(dict(
            win_id=int(row["win_id"]),
            n_triplets=int(M),
            mean_abs_dOmega=float(np.mean(np.abs(dO_vals))) if M else np.nan
        ))
        print(f"[win {w:04d}] M={M}  mean|dΩ|={summaries[-1]['mean_abs_dOmega']:.4g}")

    np.savez_compressed(
        out_root / "domega.npz",
        dOmega=np.array(dOmega_list, dtype=object),
        dOmega_SE=np.array(dOmega_SE_list, dtype=object),
        candidates_path=args.candidates,
        windows_tsv=args.windows,
        counts_path=args.counts,
        labels_path=args.labels,
        settings=json.dumps(dict(
            estimator=args.estimator, model_order=args.model_order,
            bootstrap=args.bootstrap, seed=args.seed,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )),
    )
    pd.DataFrame(summaries).to_csv(out_root / "domega_summaries.tsv", sep="\t", index=False)
    print(f"[ok] wrote {out_root/'domega.npz'} and summaries TSV")

if __name__ == "__main__":
    main()
