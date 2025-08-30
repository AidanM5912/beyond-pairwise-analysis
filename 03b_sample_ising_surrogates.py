#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
*untested

Sample equilibrium Ising surrogates per window (pairwise-only null).


Outputs:
  counts_iso.npz with keys:
    - counts_iso: list of (n,T_w) uint8 arrays per window
    - seeds: per-window seed used
    - win_bounds: (n_windows, 2) start/end bins
    - windows_tsv, counts_path, ising_path

Notes:
    - Robust counts key fallback: accept 'X' or 'counts'
    - Assert alignment between Ising win_bounds and TSV rows

"""

import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd

def gibbs_ising(J, h, T, burn_sweeps, thin, seed):
    rng = np.random.default_rng(seed)
    n = J.shape[0]
    s = rng.choice([-1, 1], size=n).astype(np.int8)
    samples = []
    total_sweeps = burn_sweeps + T * thin
    for t in range(total_sweeps):
        for i in range(n):
            field = 2.0 * (h[i] + np.dot(J[i], s))
            p = 1.0 / (1.0 + np.exp(-np.clip(field, -50.0, 50.0)))
            s[i] = 1 if rng.random() < p else -1
        if t >= burn_sweeps and ((t - burn_sweeps) % thin == 0):
            samples.append(s.copy())
    S = np.array(samples, dtype=np.int8).T
    return ((S + 1) // 2).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ising", required=True)
    ap.add_argument("--counts", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--burn-mult", type=float, default=5.0, help="burn sweeps multiplier * n")
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    wdf = pd.read_csv(args.windows, sep="\t")

    npz_iso = np.load(args.ising, allow_pickle=True)
    J_list = list(npz_iso["J_list"])
    h_list = list(npz_iso["h_list"])
    win_bounds = np.array(npz_iso["win_bounds"]) if "win_bounds" in npz_iso.files else None

    npz_cnt = np.load(args.counts, allow_pickle=True)
    ckey = "X" if "X" in npz_cnt.files else ("counts" if "counts" in npz_cnt.files else None)
    if ckey is None:
        raise ValueError(f"{args.counts} missing 'X' or 'counts'")
    Xfull = npz_cnt[ckey]
    bin_ms = float(npz_cnt["bin_ms"]) if "bin_ms" in npz_cnt.files else 10.0

    counts_list, seeds = [], []
    for w, row in wdf.iterrows():
        s, e = int(row["start_bin"]), int(row["end_bin"])
        if win_bounds is not None:
            if not (int(win_bounds[w,0]) == s and int(win_bounds[w,1]) == e):
                raise ValueError(f"Window {w}: ising bounds {tuple(win_bounds[w])} != TSV {(s,e)}")
        T = e - s
        J = J_list[w].astype(np.float64)
        h = h_list[w].astype(np.float64)
        burn = int(round(args.burn_mult * J.shape[0]))
        seed = int(args.seed + w)
        Xsim = gibbs_ising(J, h, T=T, burn_sweeps=burn, thin=args.thin, seed=seed)
        counts_list.append(Xsim)
        seeds.append(seed)
        print(f"[win {w:04d}] T={T}  seed={seed}")

    np.savez_compressed(
        out_root / "counts_iso.npz",
        counts_iso=np.array(counts_list, dtype=object),
        seeds=np.array(seeds, dtype=np.int64),
        win_bounds=win_bounds if win_bounds is not None else np.zeros((0,2), dtype=int),
        windows_tsv=args.windows,
        counts_path=args.counts,
        ising_path=args.ising,
        settings=json.dumps(dict(
            burn_mult=args.burn_mult, thin=args.thin, seed=args.seed,
            bin_ms=float(bin_ms), timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )),
    )
    print(f"[ok] wrote {out_root/'counts_iso.npz'}")

if __name__ == "__main__":
    main()
