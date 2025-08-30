#!/usr/bin/env python3
"""
01_events_from_mat.py

Ingest curated MATLAB files and write a canonical event list + metadata.

Outputs (two files, by design):
  1) <out_root>/events.npz
       - t_ms   : int32 [n_spikes]   -- spike times in *milliseconds* (rounded-to-nearest)
       - unit_id: int32 [n_spikes]   -- zero-based unit indices (stable mapping)
  2) <out_root>/meta.json
       - recording_id, duration_ms, n_units
       - source_files (list of original paths)
       - time_quantization: "round_to_nearest_ms"
       - unit_id_map (if original unit labels are non 0..N-1)
       - units_xy_path (if positions saved to .npy), units_xy_units: "micrometer"

Optionally writes:
  <out_root>/units_xy.npy  -- float32 [n_units, 2] in micrometers

Accepted MATLAB structures (any ONE is sufficient):
  A) Per-unit times:
     - 'spike_times' : cell/array, one vector of seconds per unit
     or
     - 't_spk_mat'   : cell/array, one vector of seconds per unit
  B) Pooled times + ids:
     - 'spk_times'     : vector of seconds
     - 'spk_times_id'  : vector of integer unit ids (not necessarily 0..N-1)

Unit positions (optional, same file):
  - 'xy_raw' or 'locations' : [n_units,2] or [2,n_units] (micrometers)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat


def _to_list_of_1d(arr: np.ndarray) -> List[np.ndarray]:
    """Normalize MATLAB cell/array-of-objects to a Python list of 1D float arrays."""
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return [np.asarray(arr[i].ravel(), dtype=float) for i in range(arr.size)]
    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] == 1:
        return [np.asarray(arr[0, i].ravel(), dtype=float) for i in range(arr.shape[1])]
    raise ValueError("Unsupported per-unit times structure.")


def _load_spike_times_seconds(mat: Dict) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """
    Return (per_unit_times_seconds, original_unit_ids_or_None).
    For pooled times, we return per-unit arrays and the sorted unique original unit ids.
    """
    keys = set(mat.keys())

    # Case A: per-unit times present
    if "spike_times" in keys:
        per_unit = _to_list_of_1d(mat["spike_times"])
        return per_unit, None
    if "t_spk_mat" in keys:
        per_unit = _to_list_of_1d(mat["t_spk_mat"])
        return per_unit, None

    # Case B: pooled times + ids
    if "spk_times" in keys and "spk_times_id" in keys:
        times = np.asarray(mat["spk_times"]).ravel().astype(float)
        ids = np.asarray(mat["spk_times_id"]).ravel().astype(int)
        uniq = np.unique(ids)
        per_unit = [times[ids == u] for u in uniq]
        return per_unit, uniq

    raise KeyError(
        "Could not find spike times. Expected one of: "
        "'spike_times', 't_spk_mat', or 'spk_times' + 'spk_times_id'."
    )


def _load_positions_um(mat: Dict) -> Optional[np.ndarray]:
    """Return positions [n_units,2] in micrometers, or None if not available."""
    for key in ("xy_raw", "locations"):
        if key in mat:
            arr = np.asarray(mat[key], dtype=float)
            # Normalize shape to [n,2]
            if arr.ndim == 2 and arr.shape[0] == 2:
                arr = arr.T
            if arr.ndim == 3:
                arr = arr.reshape(-1, arr.shape[-1])
            if arr.ndim == 2 and arr.shape[1] == 2:
                return arr.astype(np.float32)
    return None


def main():
    ap = argparse.ArgumentParser(description="Standardize curated MATLAB spikes to canonical events (ms) + meta.")
    ap.add_argument("--mat", required=True, help="Path to .mat file containing curated spike times.")
    ap.add_argument("--out-root", required=True, help="Output directory or prefix (e.g., outputs/Or1).")
    ap.add_argument("--recording-id", required=True, help="Recording identifier to store in metadata.")
    ap.add_argument("--force-duration-ms", type=float, default=None,
                    help="Override inferred duration (ms). If not set, uses last spike time.")
    args = ap.parse_args()

    in_path = Path(args.mat)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    mat = loadmat(in_path.as_posix(), squeeze_me=False, struct_as_record=False)

    # Times (seconds) -> per-unit lists and optional original id labels
    per_unit_s, orig_ids = _load_spike_times_seconds(mat)
    n_units = len(per_unit_s)

    # Convert to ms and build (t_ms, unit_id)
    t_ms_list = []
    unit_id_list = []
    for u, ts in enumerate(per_unit_s):
        if ts.size == 0:
            continue
        # Round to nearest ms to avoid systematic bias from floor
        ms = np.rint(ts * 1000.0).astype(np.int64)
        t_ms_list.append(ms)
        unit_id_list.append(np.full(ms.shape, u, dtype=np.int64))
    if len(t_ms_list) == 0:
        raise RuntimeError("No spikes found after parsing; cannot proceed.")

    t_ms = np.concatenate(t_ms_list).astype(np.int32)
    unit_id = np.concatenate(unit_id_list).astype(np.int32)

    # Sort by time (stable) to ensure deterministic ordering
    order = np.argsort(t_ms, kind="stable")
    t_ms = t_ms[order]
    unit_id = unit_id[order]

    # Duration
    inferred_ms = int(t_ms.max()) if t_ms.size > 0 else 0
    duration_ms = int(args.force_duration_ms) if args.force_duration_ms is not None else inferred_ms

    # Positions
    units_xy = _load_positions_um(mat)
    units_xy_path = None
    if units_xy is not None:
        if units_xy.shape[0] != n_units:
            # If original ids exist and positions align with those ids, remap explicitly
            if orig_ids is not None and units_xy.shape[0] == len(orig_ids):
                id_to_row = {int(u): i for i, u in enumerate(orig_ids.tolist())}
                remapped = np.zeros((n_units, 2), dtype=np.float32)
                for new_u, old_u in enumerate(orig_ids.tolist()):
                    remapped[new_u] = units_xy[id_to_row[old_u]]
                units_xy = remapped
            else:
                print("[warn] positions exist but length != n_units; skipping save of units_xy.")
                units_xy = None
        if units_xy is not None:
            units_xy_path = (out_root / "units_xy.npy").as_posix()
            np.save(units_xy_path, units_xy.astype(np.float32))

    # Save artifacts
    np.savez_compressed(out_root / "events.npz", t_ms=t_ms, unit_id=unit_id)

    meta = {
        "recording_id": args.recording_id,
        "duration_ms": duration_ms,
        "n_units": int(n_units),
        "source_files": [in_path.as_posix()],
        "time_quantization": "round_to_nearest_ms",
    }
    if orig_ids is not None:
        meta["unit_id_map"] = {int(i): int(j) for j, i in enumerate(orig_ids.tolist())}
    if units_xy_path is not None:
        meta["units_xy_path"] = units_xy_path
        meta["units_xy_units"] = "micrometer"

    (out_root / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[ok] events  -> {out_root/'events.npz'}   (spikes={t_ms.size}, units={n_units})")
    print(f"[ok] meta    -> {out_root/'meta.json'}")
    if units_xy_path:
        print(f"[ok] xy(μm) -> {units_xy_path}")


if __name__ == "__main__":
    main()
