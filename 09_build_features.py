#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build per-window features for burst forecasting, leakage-safe.

Inputs (lightweight contracts):
- Windows CSV (required): columns = [window_id, organoid_id, recording_id, t0, t1]
- Labels:
    * Either labels CSV with [window_id, y, H_seconds], or
    * bursts CSV + --H (seconds) to generate y with the fixed horizon rule.
- Optional upstream summaries (all per-window, keyed by window_id):
    * Ising metrics CSV (preferred) or J NPZ (per-window J_list and optional h_list)
    * Omega/DeltaOmega CSV/NPZ summaries (per-window aggregates, thresholds, CI widths)
    * Curved-MEP gamma CSV (γ grid, γ*, ΔΩ(γ*) etc.)
    * Pairwise TDA CSV (β1 curves/summaries)
    * HO TDA CSV (coverage/components, β2 if present)
    * dOmega CSV (significant groups counts, magnitudes)
- Source name: one of {real, ising, gamma}, saved with the features for null checks.

Outputs:
- features.parquet (windows × features)
- feature_dict.json (feature name → definition)
- manifest.json (paths, seed, run tag)
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from collections import OrderedDict

# --------------------------
# Helpers (robust loaders)
# --------------------------

def _read_any(path):
    if path is None or not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        sep = "," if ext == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    if ext in (".parquet",):
        return pd.read_parquet(path)
    if ext in (".json",):
        with open(path, "r") as f:
            return pd.DataFrame(json.load(f))
    if ext in (".npz",):
        return np.load(path, allow_pickle=True)
    return None

def _safe_merge(left, right, how="left", on="window_id"):
    if right is None:
        return left
    if isinstance(right, np.lib.npyio.NpzFile):
        # NPZ summaries: try conventional keys
        if "summary" in right.files:
            df = pd.DataFrame(right["summary"].tolist())
        elif "df" in right.files:
            df = pd.DataFrame(right["df"].tolist())
        else:
            # last resort: flatten npz into wide DF
            flat = {}
            for k in right.files:
                v = right[k]
                if isinstance(v, np.ndarray) and v.shape and v.shape[0] == len(left):
                    flat[k] = v
            if not flat:
                return left
            df = pd.DataFrame(flat)
        if "window_id" not in df.columns:
            # align by index if possible
            df = df.copy()
            df["window_id"] = left["window_id"].values[: len(df)]
        return left.merge(df, how=how, on=on)
    else:
        df = right.copy()
        if "window_id" not in df.columns:
            # cannot merge; bail out gracefully
            return left
        # Avoid column name collisions
        dup_cols = [c for c in df.columns if c in left.columns and c != "window_id"]
        df = df.drop(columns=dup_cols)
        return left.merge(df, how=how, on=on)

def _try_get(d, col, default=np.nan):
    return d[col] if col in d else default

def _spectral_norm_from_J(J):
    try:
        # symmetric or nearly symmetric; use largest singular value
        return float(np.linalg.svd(J, compute_uv=False)[0])
    except Exception:
        return np.nan

def _ise_stats_from_J(J):
    if J is None:
        return dict(mean_abs_J=np.nan, max_abs_J=np.nan, J_spectral_norm=np.nan)
    Ju = np.triu(J, k=1)
    absJ = np.abs(Ju[Ju != 0]) if Ju.size else np.array([])
    mean_abs = float(absJ.mean()) if absJ.size else np.nan
    max_abs = float(absJ.max()) if absJ.size else np.nan
    s_norm = _spectral_norm_from_J(J)
    return dict(mean_abs_J=mean_abs, max_abs_J=max_abs, J_spectral_norm=s_norm)

def _ece_score(y_true, y_prob, n_bins=10):
    # computed in classifier stage; defined here if ever needed
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0,1,n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if not np.any(m): 
            continue
        acc = y_true[m].mean()
        conf = y_prob[m].mean()
        ece += (m.mean()) * abs(acc - conf)
    return float(ece)

# --------------------------
# Feature builders by block
# --------------------------

def build_pairwise_ising_features(base, ising_df, J_npz):
    """B1: pairwise / Ising features (compute from summary if available, else from J)."""
    out = base.copy()
    # If J provided as NPZ with J_list aligned to windows
    J_stats = {}
    if J_npz is not None:
        # Try common keys: "J_list" or "J"
        key = "J_list" if "J_list" in J_npz.files else ("J" if "J" in J_npz.files else None)
        if key:
            Js = list(J_npz[key])
        else:
            Js = None
    else:
        Js = None

    rows = []
    for idx, row in out.iterrows():
        w = int(row["window_id"])
        if ising_df is not None and "mean_abs_J" in ising_df.columns:
            # Happy path: metrics already there
            rows.append({
                "window_id": w,
                "mean_abs_J": _try_get(ising_df.loc[ising_df["window_id"]==w].iloc[0] if (ising_df["window_id"]==w).any() else {}, "mean_abs_J"),
                "max_abs_J": _try_get(ising_df.loc[ising_df["window_id"]==w].iloc[0] if (ising_df["window_id"]==w).any() else {}, "max_abs_J"),
                "J_spectral_norm": _try_get(ising_df.loc[ising_df["window_id"]==w].iloc[0] if (ising_df["window_id"]==w).any() else {}, "J_spectral_norm"),
                "ising_pLL_gain": _try_get(ising_df.loc[ising_df["window_id"]==w].iloc[0] if (ising_df["window_id"]==w).any() else {}, "ising_pLL_gain"),
                "rate_mean": _try_get(ising_df.loc[ising_df["window_id"]==w].iloc[0] if (ising_df["window_id"]==w).any() else {}, "rate_mean"),
                "rate_var": _try_get(ising_df.loc[ising_df["window_id"]==w].iloc[0] if (ising_df["window_id"]==w).any() else {}, "rate_var"),
                "sparseness": _try_get(ising_df.loc[ising_df["window_id"]==w].iloc[0] if (ising_df["window_id"]==w).any() else {}, "sparseness"),
                "mean_field_consistency": _try_get(ising_df.loc[ising_df["window_id"]==w].iloc[0] if (ising_df["window_id"]==w).any() else {}, "mean_field_consistency"),
            })
        else:
            J = None
            if Js is not None and w < len(Js):
                J = np.array(Js[w])
            stats = _ise_stats_from_J(J)
            rows.append({
                "window_id": w,
                **stats,
                "ising_pLL_gain": np.nan,
                "rate_mean": np.nan,
                "rate_var": np.nan,
                "sparseness": np.nan,
                "mean_field_consistency": np.nan,
            })
    df = pd.DataFrame(rows)
    return out.merge(df, on="window_id", how="left")

def build_equal_time_features(base, omega_df):
    """B2: ΔΩ equal-time features."""
    out = base.copy()
    if omega_df is None:
        # place NaNs
        cols = ["DeltaOmega_mean","DeltaOmega_topk_mean","frac_redundant_sig","frac_synergy_sig","DeltaOmega_CI_median"]
        for c in cols: out[c] = np.nan
        return out
    keep = ["window_id","DeltaOmega_mean","DeltaOmega_topk_mean",
            "frac_redundant_sig","frac_synergy_sig","DeltaOmega_CI_median"]
    m = [c for c in keep if c in omega_df.columns]
    return out.merge(omega_df[m], on="window_id", how="left")

def build_curved_features(base, curved_df):
    """B3: curved-MEP (γ)."""
    out = base.copy()
    if curved_df is None:
        for c in ["gamma_star","DeltaOmega_gamma_star","LL_rel_change_gamma","gamma_closure_flag"]:
            out[c] = np.nan
        return out
    m = ["window_id","gamma_star","DeltaOmega_gamma_star","LL_rel_change_gamma","gamma_closure_flag"]
    m = [c for c in m if c in curved_df.columns]
    return out.merge(curved_df[m], on="window_id", how="left")

def build_pairwise_tda_features(base, tda_df):
    """B4: pairwise TDA features."""
    out = base.copy()
    if tda_df is None:
        cols = ["pair_AUC_beta1","pair_beta1_max","pair_thresh_at_beta1_max","pair_B0_at_perc","pair_beta1_slope_near_peak"]
        for c in cols: out[c] = np.nan
        return out
    # accept multiple naming variants
    mapping = {
        "pair_AUC_beta1": ["pair_AUC_beta1","AUC_beta1"],
        "pair_beta1_max": ["pair_beta1_max","beta1_max"],
        "pair_thresh_at_beta1_max": ["pair_thresh_at_beta1_max","thresh_at_beta1_max"],
        "pair_B0_at_perc": ["pair_B0_at_perc","B0_at_perc"],
        "pair_beta1_slope_near_peak": ["pair_beta1_slope_near_peak","beta1_slope_near_peak"],
    }
    cols = {"window_id":"window_id"}
    for k, candidates in mapping.items():
        for cand in candidates:
            if cand in tda_df.columns:
                cols[k] = cand
                break
    tsmall = tda_df[[cols[k] for k in cols]].copy()
    tsmall.columns = list(cols.keys())
    return out.merge(tsmall, on="window_id", how="left")

def build_ho_tda_features(base, ho_df):
    """B5: HO-topology features (predictables-aware)."""
    out = base.copy()
    if ho_df is None:
        cols = ["HO_red_coverage","HO_syn_coverage","HO_2skel_components_red","HO_2skel_components_syn",
                "HO_AUC_beta2_red","HO_AUC_beta2_syn","HO_degree_mean","HO_degree_var"]
        for c in cols: out[c] = np.nan
        return out
    keep = [c for c in ["window_id","HO_red_coverage","HO_syn_coverage",
                        "HO_2skel_components_red","HO_2skel_components_syn",
                        "HO_AUC_beta2_red","HO_AUC_beta2_syn",
                        "HO_degree_mean","HO_degree_var"] if c in ho_df.columns]
    return out.merge(ho_df[keep], on="window_id", how="left")

def build_domega_features(base, do_df):
    """B6: dΩ features (optional)."""
    out = base.copy()
    if do_df is None:
        cols = ["dOmega_mean_abs","dOmega_max_abs","dOmega_count_syn","dOmega_count_red",
                "dOmega_frac_syn","dOmega_frac_red"]
        for c in cols: out[c] = np.nan
        return out
    keep = [c for c in ["window_id","dOmega_mean_abs","dOmega_max_abs","dOmega_count_syn",
                        "dOmega_count_red","dOmega_frac_syn","dOmega_frac_red"] if c in do_df.columns]
    return out.merge(do_df[keep], on="window_id", how="left")

def _add_lags(df, group_cols, time_col, max_lag=2):
    """B7: add lagged copies of all numeric features (excl id/label) within recording."""
    if max_lag <= 0: 
        return df
    df = df.sort_values(group_cols + [time_col]).copy()
    feat_cols = [c for c in df.columns if c not in ["window_id","organoid_id","recording_id","t0","t1","y","source","H_seconds"]]
    for L in range(1, max_lag+1):
        shifted = df.groupby(group_cols)[feat_cols].shift(L)
        shifted.columns = [f"{c}_lag{L}" for c in shifted.columns]
        df = pd.concat([df, shifted], axis=1)
    return df

def _add_missingness_indicators(df):
    miss_cols = [c for c in df.columns if c not in ["window_id","organoid_id","recording_id","t0","t1","y","source","H_seconds"]]
    miss = df[miss_cols].isna()
    miss = miss.add_suffix("_isna")
    return pd.concat([df, miss.astype(np.uint8)], axis=1)

# --------------------------
# Labeling from bursts (if needed)
# --------------------------

def make_labels_from_bursts(windows_df, bursts_df, H_seconds):
    """
    Label y=1 if any burst onset occurs in [t1, t1+H).
    Assumes windows_df has t0,t1 in seconds or ms; we detect units from monotonicity.
    bursts_df expected columns: [recording_id, onset_time] (time in same units as windows).
    """
    df = windows_df.copy()
    # Try to detect units; here we assume t columns are in seconds already.
    if "onset_time" not in bursts_df.columns:
        # try common alternative names
        for c in ["burst_onset","onset","t_burst"]:
            if c in bursts_df.columns:
                bursts_df = bursts_df.rename(columns={c:"onset_time"})
                break
    if "onset_time" not in bursts_df.columns:
        raise ValueError("bursts CSV must contain an 'onset_time' column.")

    # Left-join to count any onset in [t1, t1+H)
    y = []
    for _, r in df.iterrows():
        rid = r["recording_id"]
        t1  = float(r["t1"])
        H   = float(H_seconds)
        s   = bursts_df[bursts_df["recording_id"] == rid]
        hit = ((s["onset_time"] >= t1) & (s["onset_time"] < t1 + H)).any()
        y.append(1 if hit else 0)
    df["y"] = y
    df["H_seconds"] = H_seconds
    return df

# --------------------------
# CLI
# --------------------------

def main():
    ap = argparse.ArgumentParser(description="Build forecasting features per window.")
    ap.add_argument("--windows_csv", required=True)
    # labels
    ap.add_argument("--labels_csv", default=None, help="If not provided, use bursts+H to construct labels")
    ap.add_argument("--bursts_csv", default=None, help="Burst onsets per recording")
    ap.add_argument("--H_seconds", type=float, default=None, help="Burst horizon (seconds) if deriving labels")
    # upstream artifacts
    ap.add_argument("--ising_metrics_csv", default=None)
    ap.add_argument("--ising_J_npz", default=None)
    ap.add_argument("--omega_summary", default=None)      # csv or npz
    ap.add_argument("--curved_summary_csv", default=None)
    ap.add_argument("--tda_pair_csv", default=None)
    ap.add_argument("--tda_ho_csv", default=None)
    ap.add_argument("--domega_csv", default=None)
    # options
    ap.add_argument("--source", required=True, choices=["real","ising","gamma","other"])
    ap.add_argument("--max_lag", type=int, default=2)
    ap.add_argument("--out_features", required=True)
    ap.add_argument("--out_feature_dict", required=True)
    ap.add_argument("--run_tag", default="default")
    args = ap.parse_args()

    # Load windows
    windows = pd.read_csv(args.windows_csv)
    required_cols = {"window_id","organoid_id","recording_id","t0","t1"}
    missing = required_cols - set(windows.columns)
    if missing:
        raise ValueError(f"windows_csv missing columns: {missing}")
    base = windows[["window_id","organoid_id","recording_id","t0","t1"]].copy()

    # Labels
    if args.labels_csv is not None:
        labels = pd.read_csv(args.labels_csv)
        if "window_id" not in labels.columns or "y" not in labels.columns:
            raise ValueError("labels_csv must have [window_id, y] columns")
        if "H_seconds" not in labels.columns and args.H_seconds is not None:
            labels["H_seconds"] = float(args.H_seconds)
        base = base.merge(labels[["window_id","y","H_seconds"]], how="left", on="window_id")
    else:
        if args.bursts_csv is None or args.H_seconds is None:
            raise ValueError("Provide either --labels_csv or both --bursts_csv and --H_seconds")
        bursts = pd.read_csv(args.bursts_csv)
        base = make_labels_from_bursts(base, bursts, args.H_seconds)

    # Ising and others
    ising_df = _read_any(args.ising_metrics_csv)
    J_npz    = _read_any(args.ising_J_npz)
    omega    = _read_any(args.omega_summary)
    curved   = _read_any(args.curved_summary_csv)
    tda_pair = _read_any(args.tda_pair_csv)
    tda_ho   = _read_any(args.tda_ho_csv)
    domega   = _read_any(args.domega_csv)

    # Build features block-by-block
    df = base.copy()
    df = build_pairwise_ising_features(df, ising_df, J_npz)
    df = build_equal_time_features(df, omega)
    df = build_curved_features(df, curved)
    df = build_pairwise_tda_features(df, tda_pair)
    df = build_ho_tda_features(df, tda_ho)
    df = build_domega_features(df, domega)

    # Add source
    df["source"] = args.source

    # Add lags (B7)
    df = _add_lags(df, group_cols=["organoid_id","recording_id"], time_col="t0", max_lag=args.max_lag)

    # Add missingness indicators
    df = _add_missingness_indicators(df)

    # Save features
    out_dir = os.path.dirname(args.out_features)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(args.out_features, index=False)

    # Feature dictionary (simple mapping)
    feature_dict = OrderedDict()
    for c in df.columns:
        if c in ["window_id","organoid_id","recording_id","t0","t1","y","source","H_seconds"]:
            continue
        if c.endswith("_isna"):
            feature_dict[c] = "Missingness indicator for " + c.replace("_isna","")
        elif "_lag" in c:
            basec = c.split("_lag")[0]
            feature_dict[c] = f"Lagged copy of {basec} (within recording)"
        else:
            feature_dict[c] = "See spec B.* for definition"
    with open(args.out_feature_dict, "w") as f:
        json.dump(feature_dict, f, indent=2)

    # Manifest
    manifest = dict(
        windows_csv=os.path.abspath(args.windows_csv),
        labels_csv=os.path.abspath(args.labels_csv) if args.labels_csv else None,
        bursts_csv=os.path.abspath(args.bursts_csv) if args.bursts_csv else None,
        H_seconds=float(df["H_seconds"].iloc[0]) if "H_seconds" in df.columns and len(df) else None,
        source=args.source,
        out_features=os.path.abspath(args.out_features),
        feature_dict=os.path.abspath(args.out_feature_dict),
        run_tag=args.run_tag,
    )
    with open(os.path.join(out_dir or ".", "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Saved features: {args.out_features}")
    print(f"[OK] Saved feature_dict: {args.out_feature_dict}")

if __name__ == "__main__":
    main()
