#!/usr/bin/env python3
"""
*untested

02_make_windows.py

Bin canonical events (ms) to counts and create overlapping feature windows.

Inputs:
  <in-root>/events.npz   -- t_ms:int32, unit_id:int32
  <in-root>/meta.json    -- duration_ms (optional but recommended)

Outputs:
  1) <out_root>/counts_bin{bin_ms}ms.npz
       - X        : uint16 [n_units, n_bins]  (spike counts per bin)
       - bin_ms   : float
       - duration_ms : int
  2) <out_root>/windows_win{win_s}s_ov{overlap}.tsv                (all windows)
  3) <out_root>/windows_win{win_s}s_ov{overlap}.filtered.tsv       (FR-drift ≤ 15%)
       Columns: win_id, start_bin, end_bin, start_s, end_s
  4) <out_root>/windows_win{...}.meta.json
       - n_windows, kept_windows, dropped_windows
       - win_s, overlap, bin_ms, n_bins, duration_ms, recording_id
       - qc: { fr_drift_pct_median, drift_threshold_pct }
       - windows_tsv, windows_filtered_tsv

Notes:
  - Default Δt = 10 ms (spec primary). Use --enforce-spec to require Δt=10 ms,
    Tw ∈ {1.0, 2.0} s, overlap = 0.5, and acceptance: ≥100 bins & ≥20 blocks @ 50 ms.
  - Stationarity gating: drop windows with FR drift > 15% (between halves).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def _compute_counts(n_units: int, duration_ms: int, bin_ms: float, t_ms: np.ndarray, unit_id: np.ndarray) -> np.ndarray:
    """Efficient binning of events to a [n_units, n_bins] uint16 matrix."""
    bin_ms = float(bin_ms)
    n_bins = int(np.ceil(duration_ms / bin_ms))
    X = np.zeros((n_units, n_bins), dtype=np.uint16)

    # Bin indices (clip to range)
    b = np.floor(t_ms / bin_ms).astype(np.int64)
    b = np.clip(b, 0, n_bins - 1)

    # Aggregate using numpy add.at for sparse accumulation
    np.add.at(X, (unit_id, b), 1)
    return X


def main():
    ap = argparse.ArgumentParser(description="Bin event list and create overlapping feature windows.")
    ap.add_argument("--in-root", required=True, help="Directory containing events.npz and meta.json")
    ap.add_argument("--out-root", required=True, help="Destination directory for outputs")
    ap.add_argument("--bin-ms", type=float, default=10.0,
                    help="Bin width in milliseconds (primary=10; sensitivity=5 or 20)")
    ap.add_argument("--win-s", type=float, default=1.0, help="Window length in seconds (e.g., 1.0 or 2.0)")
    ap.add_argument("--overlap", type=float, default=0.5, help="Fractional overlap (must be 0.5 for acceptance).")
    ap.add_argument("--enforce-spec", action="store_true",
                    help="Fail if bin-ms != 10, win-s not in {1.0, 2.0}, or overlap != 0.5. "
                         "Also enforces ≥100 bins & ≥20×50ms blocks per window.")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load events + meta
    ev = np.load(in_root / "events.npz")
    t_ms = ev["t_ms"].astype(np.int64)
    unit_id = ev["unit_id"].astype(np.int64)
    meta = json.loads((in_root / "meta.json").read_text())
    duration_ms = int(meta.get("duration_ms", int(t_ms.max()) if t_ms.size else 0))
    n_units = int(meta.get("n_units", int(unit_id.max()) + 1))
    recording_id = meta.get("recording_id", "unknown")

    # Spec enforcement checks
    if args.enforce-spec:
        if abs(args.bin_ms - 10.0) > 1e-9:
            raise ValueError("Spec violation: bin-ms must be 10.0 for acceptance analyses.")
        if args.win_s not in (1.0, 2.0):
            raise ValueError("Spec violation: win-s must be 1.0 or 2.0 for acceptance analyses.")
        if abs(args.overlap - 0.5) > 1e-9:
            raise ValueError("Spec violation: overlap must be 0.5.")

    # Bin to counts
    X = _compute_counts(n_units, duration_ms, args.bin_ms, t_ms, unit_id)
    counts_path = out_root / f"counts_bin{int(args.bin_ms)}ms.npz"
    np.savez_compressed(
        counts_path,
        X=X, bin_ms=float(args.bin_ms), duration_ms=int(duration_ms)
    )
    print(f"[ok] counts -> {counts_path} (units={X.shape[0]}, bins={X.shape[1]}, bin_ms={args.bin_ms})")

    # Make overlapping feature windows
    bin_s = args.bin_ms / 1000.0
    n_bins = X.shape[1]
    win_bins = max(1, int(round(args.win_s / bin_s)))
    step_bins = max(1, int(round(args.win_s * (1.0 - args.overlap) / bin_s)))

    if args.enforce-spec and win_bins < 100:
        raise ValueError(f"Spec violation: window bins = {win_bins} < 100 "
                         f"(bin_ms={args.bin_ms}, win_s={args.win_s}).")
    if args.enforce-spec:
        n_blocks = (args.win_s / 0.050)
        if n_blocks < 20 - 1e-9:
            raise ValueError(f"Spec violation: bootstrap blocks = {n_blocks:.1f} < 20 (win_s={args.win_s}).")

    starts = np.arange(0, max(0, n_bins - win_bins + 1), step_bins, dtype=int)
    ends = starts + win_bins

    # Stationarity gating: drop windows with FR drift > 15%
    def pop_fr(W):
        return float(W.sum()) / (W.shape[1] * bin_s)

    keep_mask, drifts = [], []
    for s, e in zip(starts, ends):
        W = X[:, s:e]
        half = W.shape[1] // 2
        if half == 0:
            keep = False
            drift_pct = None
        else:
            fr1, fr2 = pop_fr(W[:, :half]), pop_fr(W[:, half:])
            drift_pct = None if (fr1 + fr2) == 0 else abs(fr2 - fr1) / ((fr1 + fr2) / 2.0) * 100.0
            keep = (drift_pct is None) or (drift_pct <= 15.0)
        keep_mask.append(keep)
        drifts.append(drift_pct)

    kept_idx = np.where(keep_mask)[0]

    # Write all windows TSV and filtered TSV
    tag = f"win{args.win_s:g}s_ov{int(args.overlap*100)}"
    wfile = out_root / f"windows_{tag}.tsv"
    wfilt = out_root / f"windows_{tag}.filtered.tsv"

    with wfile.open("w") as f:
        f.write("win_id\tstart_bin\tend_bin\tstart_s\tend_s\n")
        for i, (s, e) in enumerate(zip(starts, ends)):
            f.write(f"{i}\t{s}\t{e}\t{s*bin_s:.6f}\t{e*bin_s:.6f}\n")

    with wfilt.open("w") as f:
        f.write("win_id\tstart_bin\tend_bin\tstart_s\tend_s\n")
        for i in kept_idx:
            s, e = starts[i], ends[i]
            f.write(f"{i}\t{s}\t{e}\t{s*bin_s:.6f}\t{e*bin_s:.6f}\n")

    print(f"[ok] windows (all)  -> {wfile} (n_windows={len(starts)})")
    print(f"[ok] windows (kept) -> {wfilt} (kept={len(kept_idx)}, dropped={len(starts)-len(kept_idx)})")

    # Metadata with QC and recording_id
    drift_vals = [d for d in drifts if d is not None]
    wmeta = {
        "recording_id": recording_id,
        "n_windows": int(len(starts)),
        "kept_windows": int(len(kept_idx)),
        "dropped_windows": int(len(starts) - len(kept_idx)),
        "win_s": float(args.win_s),
        "overlap": float(args.overlap),
        "bin_ms": float(args.bin_ms),
        "n_bins": int(n_bins),
        "duration_ms": int(duration_ms),
        "qc": {
            "fr_drift_pct_median": float(np.nanmedian(drift_vals)) if drift_vals else None,
            "drift_threshold_pct": 15.0
        },
        "windows_tsv": wfile.as_posix(),
        "windows_filtered_tsv": wfilt.as_posix(),
    }
    (out_root / f"windows_{tag}.meta.json").write_text(json.dumps(wmeta, indent=2))


if __name__ == "__main__":
    main()
