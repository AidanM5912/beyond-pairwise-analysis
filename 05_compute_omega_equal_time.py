#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
*untested

Equal-time Ω and ΔΩ between REAL and a SURROGATE (Ising or curved).

Inputs
  --counts     counts_bin10ms.npz  (keys: X or counts; must hold bin_ms)
  --surrogate  counts_iso.npz OR gamma_*/counts_curved.npz
  --windows    windows_*_ov50.filtered.tsv
  --candidates candidates.npz  (triangles_per_window: list of (M,3) 0-based)
  --out-root   output directory
  --estimator  gcmi|lin_est (default gcmi)
  --bootstrap  block bootstrap reps (default 100)
  --block-ms   block length in ms (default 50)
  --seed       RNG seed

Outputs (NPZ)
  - Omega_real: list per window (float array per triplet)
  - Omega_surr: list per window
  - DeltaOmega_iso: list per window (Ω_real - Ω_surr)
  - se_Omega_real, se_Delta: list per window (bootstrap SEs)
  - surrogate_path (string), surrogate_key

Notes:
- Robust counts key fallback: accept 'X' or 'counts'
- Surrogate key fallback: accept 'counts_iso' (Ising) OR 'counts_curved' (γ-surrogates)
- Case-correct import (toolbox.oinfo)
- Persist surrogate_path in the NPZ so 8c can recover the Ising path for the shuffled baseline.

"""

import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from toolbox.gcmi import copnorm
from toolbox.oinfo import o_information_boot

def block_bootstrap_indices(T, block, B, rng):
    starts = np.arange(0, T - block + 1, block, dtype=int)
    if len(starts) == 0:
        starts = np.array([0], dtype=int)
    idxs = []
    for _ in range(B):
        picks = rng.integers(0, len(starts), size=len(starts), endpoint=False)
        arr = []
        for p in picks:
            arr.extend(range(starts[p], min(starts[p] + block, T)))
        idxs.append(np.array(arr[:T], dtype=int))
    return idxs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--surrogate", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--estimator", choices=["gcmi","lin_est"], default="gcmi")
    ap.add_argument("--bootstrap", type=int, default=100)
    ap.add_argument("--block-ms", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=2024)
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

    npz_sur = np.load(args.surrogate, allow_pickle=True)
    skey = "counts_iso" if "counts_iso" in npz_sur.files else ("counts_curved" if "counts_curved" in npz_sur.files else None)
    if skey is None:
        raise ValueError("--surrogate npz must have 'counts_iso' or 'counts_curved'")
    S_list = list(npz_sur[skey])

    Omega_real, Omega_surr, Delta, se_OmR, se_Dl = [], [], [], [], []

    for w, row in tqdm(list(wdf.iterrows()), desc="Windows"):
        s, e = int(row["start_bin"]), int(row["end_bin"])
        Xr = Xfull[:, s:e]
        Xs = np.asarray(S_list[w], dtype=np.float64)

        # estimator parity: copula-normalize both
        Xr_c = copnorm(Xr)
        Xs_c = copnorm(Xs)

        T = Xr.shape[1]
        block = max(1, int(round(args.block_ms / bin_ms)))
        tri = np.asarray(tri_list[w], dtype=np.int32)
        ind_all = np.arange(T, dtype=np.int32)

        OmR = np.empty(len(tri), dtype=np.float64)
        OmS = np.empty(len(tri), dtype=np.float64)
        Dl  = np.empty(len(tri), dtype=np.float64)
        seR = np.empty(len(tri), dtype=np.float64)
        seD = np.empty(len(tri), dtype=np.float64)

        boot_idxs = block_bootstrap_indices(T, block, args.bootstrap, rng)

        for m, (i,j,k) in enumerate(tri):
            indv = np.array([i,j,k], dtype=np.int32)
            try:
                oR = float(o_information_boot(Xr_c, ind_all, indv, args.estimator))
                oS = float(o_information_boot(Xs_c, ind_all, indv, args.estimator))
            except np.linalg.LinAlgError:
                # tiny jitter fallback
                eps = 1e-6
                oR = float(o_information_boot(Xr_c + eps*np.random.default_rng(0).normal(size=Xr_c.shape), ind_all, indv, args.estimator))
                oS = float(o_information_boot(Xs_c + eps*np.random.default_rng(1).normal(size=Xs_c.shape), ind_all, indv, args.estimator))
            OmR[m] = oR; OmS[m] = oS; Dl[m] = oR - oS

            # bootstrap SEs
            br = []
            bd = []
            for idx in boot_idxs:
                oRb = float(o_information_boot(Xr_c, idx, indv, args.estimator))
                oSb = float(o_information_boot(Xs_c, idx, indv, args.estimator))
                br.append(oRb)
                bd.append(oRb - oSb)
            seR[m] = float(np.std(br, ddof=1))
            seD[m] = float(np.std(bd, ddof=1))

        Omega_real.append(OmR)
        Omega_surr.append(OmS)
        Delta.append(Dl)
        se_OmR.append(seR)
        se_Dl.append(seD)

    np.savez_compressed(
        out_root / "omega_equal_time.npz",
        Omega_real=np.array(Omega_real, dtype=object),
        Omega_surr=np.array(Omega_surr, dtype=object),
        DeltaOmega_iso=np.array(Delta, dtype=object),
        se_Omega_real=np.array(se_OmR, dtype=object),
        se_Delta=np.array(se_Dl, dtype=object),
        windows_tsv=args.windows,
        counts_path=args.counts,
        surrogate_path=args.surrogate,  # <— used by 08c to recover Ising path
        surrogate_key=skey,
        candidates_path=args.candidates,
        settings=json.dumps(dict(
            estimator=args.estimator, bootstrap=args.bootstrap, block_ms=args.block_ms,
            bin_ms=float(bin_ms), seed=args.seed,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )),
    )
    print(f"[ok] wrote {out_root/'omega_equal_time.npz'}")

if __name__ == "__main__":
    main()
