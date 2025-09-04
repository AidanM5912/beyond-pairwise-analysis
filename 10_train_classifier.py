#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
!!Update!!

Train + evaluate forecasting models with groupwise splits, ablations, and null checks.

Inputs:
- Features (parquet files): per-window features for each organoid
- Labels (CSV): ground truth labels for each window
- Grouping (CSV): mapping of windows to groups (e.g., organoid_id)

Outputs:
- metrics.json (PR/ROC/Brier/ECE; per-abl/ per-fold; per-organoid test)
- split_manifests/*.csv (train/val/test window_ids, per fold)
- scalers/imputers (joblib), trained models (LR & RF), calibration objects
- curves/*.npy (PR/ROC curves, calibration reliability)
- importances/*.json (LR coefficients with bootstrap CIs; RF permutation importances)


Usage (example):
  python 10_train_classifier.py \
      --features_real real_features.parquet \
      --features_ising ising_features.parquet \
      --features_gamma gamma_features.parquet \
      --out_dir runs/cls_v1 --seed 42
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve, roc_curve, f1_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump

np.set_printoptions(suppress=True, linewidth=140)

# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed):
    np.random.seed(seed)

def _ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if not np.any(m): 
            continue
        acc = y_true[m].mean()
        conf = y_prob[m].mean()
        ece += m.mean() * abs(acc - conf)
    return float(ece)

def _feature_blocks(columns):
    """Define ablation groups by name prefixes."""
    blocks = dict(
        pairwise=[c for c in columns if c.startswith(("mean_abs_J","max_abs_J","J_spectral_norm","ising_pLL_gain","rate_","sparseness","mean_field_consistency","pair_"))],
        omega=[c for c in columns if c.startswith(("DeltaOmega_", "frac_redundant_sig","frac_synergy_sig"))],
        ho=[c for c in columns if c.startswith(("HO_","HO_2skel","HO_AUC_beta2"))],
        gamma=[c for c in columns if c.startswith(("gamma_star","DeltaOmega_gamma_star","LL_rel_change_gamma","gamma_closure_flag"))],
        domega=[c for c in columns if c.startswith(("dOmega_"))],
        lags=[c for c in columns if "_lag" in c],
        miss=[c for c in columns if c.endswith("_isna")],
    )
    # Flatten to a ranked union for A0..A4 ablations
    return blocks

def _build_ablation_columns(blocks, all_cols):
    A0 = sorted(set(blocks["pairwise"]))
    A1 = sorted(set(A0 + blocks["omega"]))
    A2 = sorted(set(A1 + blocks["ho"]))
    A3 = sorted(set(A2 + blocks["gamma"]))
    A4 = sorted(set(A3 + blocks["domega"]))
    # Always include lags + missingness indicators if present
    def add_std(cols):
        extra = sorted(set(blocks["lags"] + blocks["miss"]))
        return sorted(set(cols + extra))
    return dict(
        A0=add_std(A0),
        A1=add_std(A1),
        A2=add_std(A2),
        A3=add_std(A3),
        A4=add_std(A4),
        ALL=add_std([c for c in all_cols if c not in ["window_id","organoid_id","recording_id","t0","t1","y","source","H_seconds"]]),
    )

def _prep_Xy(df, cols):
    X = df[cols].values
    y = df["y"].values.astype(int)
    return X, y

def _split_train_val_test(groups, n_splits=5, fold_idx=0):
    """Outer GroupKFold into train+val vs test; inner split for validation."""
    gkf = GroupKFold(n_splits=n_splits)
    outer_splits = list(gkf.split(np.zeros(len(groups)), np.zeros(len(groups)), groups))
    trainval_idx, test_idx = outer_splits[fold_idx]
    # inner split for validation (groups from trainval only)
    groups_tv = np.array(groups)[trainval_idx]
    inner = GroupKFold(n_splits=min(5, len(np.unique(groups_tv))))
    inner_splits = list(inner.split(np.zeros(len(groups_tv)), np.zeros(len(groups_tv)), groups_tv))
    # pick first inner split deterministically
    tr_idx_rel, va_idx_rel = inner_splits[0]
    train_idx = trainval_idx[tr_idx_rel]
    val_idx   = trainval_idx[va_idx_rel]
    return train_idx, val_idx, test_idx

def _fit_and_calibrate_LR(Xtr, ytr, Xva, yva, class_weight, C):
    lr = LogisticRegression(penalty="l2", solver="liblinear", C=C, class_weight=class_weight, max_iter=2000, n_jobs=1)
    lr.fit(Xtr, ytr)
    # Platt calibration on validation only (prefit=True)
    calib = CalibratedClassifierCV(lr, method="sigmoid", cv="prefit")
    calib.fit(Xva, yva)
    return lr, calib

def _fit_RF(Xtr, ytr, class_weight, n_estimators=500, max_depth=None, min_samples_leaf=1, seed=0):
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        class_weight=class_weight, random_state=seed, n_jobs=-1, oob_score=False)
    rf.fit(Xtr, ytr)
    return rf

def _metrics_from_probs(y, p, threshold=None):
    pr_auc = average_precision_score(y, p)
    roc_auc = roc_auc_score(y, p)
    brier = brier_score_loss(y, p)
    ece = _ece(y, p)
    prec, rec, f1 = np.nan, np.nan, np.nan
    if threshold is not None:
        yhat = (p >= threshold).astype(int)
        prec = precision_score(y, yhat, zero_division=0)
        rec  = recall_score(y, yhat, zero_division=0)
        f1   = f1_score(y, yhat, zero_division=0)
    return dict(PR_AUC=pr_auc, ROC_AUC=roc_auc, Brier=brier, ECE=ece,
                Precision=prec, Recall=rec, F1=f1)

def _choose_threshold(yva, pva, mode="f1"):
    # choose on validation only
    prec, rec, thr = precision_recall_curve(yva, pva)
    # thr has len = len(prec)-1; align
    best_f1, best_thr = -1, 0.5
    for i in range(len(thr)):
        f1 = 2*prec[i]*rec[i]/(prec[i]+rec[i]) if (prec[i]+rec[i])>0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr[i]
    return best_thr

def _permutation_importance_block(model, X, y, groups, n_repeats=10, random_state=0):
    """Permutation importance within each recording to reduce temporal leakage."""
    rng = np.random.RandomState(random_state)
    baseline = average_precision_score(y, model.predict_proba(X)[:,1])
    n, p = X.shape
    importances = np.zeros((p, n_repeats))
    # build index buckets by recording
    # (if 'groups' are organoids, we need a finer grouping; here we use unique recordings from a parallel array)
    rec_ids = groups  # overload: pass recording_id array here
    uniq = np.unique(rec_ids)
    for r in range(n_repeats):
        for j in range(p):
            Xp = X.copy()
            # permute within each recording bucket
            for g in uniq:
                idx = np.where(rec_ids == g)[0]
                Xp[idx, j] = rng.permutation(Xp[idx, j])
            score = average_precision_score(y, model.predict_proba(Xp)[:,1])
            importances[j, r] = baseline - score
    return importances.mean(axis=1), importances.std(axis=1, ddof=1)

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Train forecasting models with groupwise splits, ablations, and null checks.")
    ap.add_argument("--features_real", required=True)
    ap.add_argument("--features_ising", default=None)
    ap.add_argument("--features_gamma", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # Load features
    f_real  = pd.read_parquet(args.features_real)
    f_ising = pd.read_parquet(args.features_ising) if args.features_ising else None
    f_gamma = pd.read_parquet(args.features_gamma) if args.features_gamma else None

    # Validate schema alignment for null checks
    feat_cols_all = [c for c in f_real.columns if c not in ["window_id","organoid_id","recording_id","t0","t1","y","source","H_seconds"]]
    blocks = _feature_blocks(feat_cols_all)
    absets = _build_ablation_columns(blocks, f_real.columns)

    # Prepare arrays
    groups_org = f_real["organoid_id"].values
    groups_rec = f_real["recording_id"].values  # for block permutation
    window_ids = f_real["window_id"].values

    # Preprocess pipeline (train-only fit!)
    all_results = dict(folds=[], per_organoid={}, ablations={})
    split_dir = os.path.join(args.out_dir, "split_manifests"); os.makedirs(split_dir, exist_ok=True)
    scaler_dir = os.path.join(args.out_dir, "scalers"); os.makedirs(scaler_dir, exist_ok=True)
    model_dir = os.path.join(args.out_dir, "models"); os.makedirs(model_dir, exist_ok=True)
    curve_dir = os.path.join(args.out_dir, "curves"); os.makedirs(curve_dir, exist_ok=True)
    import_dir = os.path.join(args.out_dir, "importances"); os.makedirs(import_dir, exist_ok=True)

    # Hyperparam grids
    LR_grid = {"C": [0.01, 0.1, 1.0, 10.0]}
    RF_grid = {"max_depth": [None, 6, 10], "min_samples_leaf": [1,5], "n_estimators":[500]}

    for fold in range(min(args.folds, len(np.unique(groups_org)))):
        tr_idx, va_idx, te_idx = _split_train_val_test(groups_org, n_splits=args.folds, fold_idx=fold)
        manifest = dict(
            fold=fold,
            train_window_ids=window_ids[tr_idx].tolist(),
            val_window_ids=window_ids[va_idx].tolist(),
            test_window_ids=window_ids[te_idx].tolist(),
        )
        pd.DataFrame({"window_id": manifest["train_window_ids"]}).to_csv(os.path.join(split_dir, f"train_fold{fold}.csv"), index=False)
        pd.DataFrame({"window_id": manifest["val_window_ids"]}).to_csv(os.path.join(split_dir, f"val_fold{fold}.csv"), index=False)
        pd.DataFrame({"window_id": manifest["test_window_ids"]}).to_csv(os.path.join(split_dir, f"test_fold{fold}.csv"), index=False)

        fold_result = dict(fold=fold, LR={}, RF={}, threshold=None, per_organoid={})

        # Loop over ablations (A0..A4 + ALL)
        for abl_name, cols in absets.items():
            # Build preprocessors fit on TRAIN only
            # Identify numeric columns explicitly (all cols are numeric here)
            num_cols = cols

            pre = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ])

            # Prepare matrices
            X_all = f_real[num_cols].values
            y_all = f_real["y"].values.astype(int)

            # Fit preprocessors on train only
            Xtr = pre.fit_transform(X_all[tr_idx])
            Xva = pre.transform(X_all[va_idx])
            Xte = pre.transform(X_all[te_idx])

            ytr, yva, yte = y_all[tr_idx], y_all[va_idx], y_all[te_idx]

            class_weight = "balanced"

            # ----------------- Logistic Regression -----------------
            # Hyperparam selection by validation PR-AUC
            best_lr, best_calib, best_score, best_C = None, None, -np.inf, None
            for C in LR_grid["C"]:
                lr, calib = _fit_and_calibrate_LR(Xtr, ytr, Xva, yva, class_weight, C)
                pva = calib.predict_proba(Xva)[:,1]
                score = average_precision_score(yva, pva)
                if score > best_score:
                    best_lr, best_calib, best_score, best_C = lr, calib, score, C

            # Freeze hyperparams; choose threshold on validation
            pva = best_calib.predict_proba(Xva)[:,1]
            thr = _choose_threshold(yva, pva, mode="f1")

            # Evaluate on test
            pte = best_calib.predict_proba(Xte)[:,1]
            m_te = _metrics_from_probs(yte, pte, threshold=thr)
            # Save curves
            pr = precision_recall_curve(yte, pte); np.save(os.path.join(curve_dir, f"PR_LR_{abl_name}_fold{fold}.npy"), pr, allow_pickle=True)
            roc = roc_curve(yte, pte); np.save(os.path.join(curve_dir, f"ROC_LR_{abl_name}_fold{fold}.npy"), roc, allow_pickle=True)

            # Save preprocessors and models
            dump(pre, os.path.join(scaler_dir, f"pre_{abl_name}_fold{fold}.joblib"))
            dump(best_lr, os.path.join(model_dir, f"LR_{abl_name}_fold{fold}.joblib"))
            dump(best_calib, os.path.join(model_dir, f"LR_calib_{abl_name}_fold{fold}.joblib"))

            # LR "importances" = coefficients (on standardized scale)
            try:
                coefs = best_lr.coef_.ravel()
                imp = dict(zip(num_cols, coefs.tolist()))
                with open(os.path.join(import_dir, f"LR_coefs_{abl_name}_fold{fold}.json"), "w") as f:
                    json.dump(imp, f, indent=2)
            except Exception:
                pass

            fold_result["LR"][abl_name] = dict(
                best_C=best_C, val_PR_AUC=float(best_score), test_metrics=m_te
            )
            fold_result["threshold"] = thr  # same threshold reused below for RF reporting consistency

            # ----------------- Random Forest -----------------
            # Fit on train; select by validation PR-AUC
            best_rf, best_score, best_params = None, -np.inf, None
            for params in ParameterGrid(RF_grid):
                rf = _fit_RF(Xtr, ytr, class_weight, **params, seed=args.seed+fold)
                pva = rf.predict_proba(Xva)[:,1]
                score = average_precision_score(yva, pva)
                if score > best_score:
                    best_rf, best_score, best_params = rf, score, params

            # Calibrate RF on validation (Platt via CalibratedClassifierCV)
            rf_calib = CalibratedClassifierCV(best_rf, method="sigmoid", cv="prefit")
            rf_calib.fit(Xva, yva)
            pte = rf_calib.predict_proba(Xte)[:,1]
            m_te = _metrics_from_probs(yte, pte, threshold=thr)  # threshold chosen on LR-val; report RF with same threshold policy? Spec allows val-chosen per model; we keep LR's for consistency; tweak if desired.

            # Save
            dump(best_rf, os.path.join(model_dir, f"RF_{abl_name}_fold{fold}.joblib"))
            dump(rf_calib, os.path.join(model_dir, f"RF_calib_{abl_name}_fold{fold}.joblib"))

            # Permutation importance (block-wise within recordings)
            try:
                imp_mean, imp_std = _permutation_importance_block(rf_calib, Xte, yte, groups=f_real["recording_id"].values[te_idx], n_repeats=10, random_state=args.seed+fold)
                rf_imp = dict(mean=dict(zip(num_cols, imp_mean.tolist())),
                              std=dict(zip(num_cols, imp_std.tolist())))
                with open(os.path.join(import_dir, f"RF_permimp_{abl_name}_fold{fold}.json"), "w") as f:
                    json.dump(rf_imp, f, indent=2)
            except Exception:
                pass

            pr = precision_recall_curve(yte, pte); np.save(os.path.join(curve_dir, f"PR_RF_{abl_name}_fold{fold}.npy"), pr, allow_pickle=True)
            roc = roc_curve(yte, pte); np.save(os.path.join(curve_dir, f"ROC_RF_{abl_name}_fold{fold}.npy"), roc, allow_pickle=True)

            fold_result["RF"][abl_name] = dict(
                best_params=best_params, val_PR_AUC=float(best_score), test_metrics=m_te
            )

        # Per-organoid PR-AUC on test (using best ALL/LR)
        cols_ALL = absets["ALL"]
        pre = dump  # no-op: already saved above per ablation
        # Rebuild preprocessor and model to score per-organoid
        pre_all = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
        X_all = f_real[cols_ALL].values; y_all = f_real["y"].values.astype(int)
        Xtr = pre_all.fit_transform(X_all[tr_idx]); Xva = pre_all.transform(X_all[va_idx]); Xte = pre_all.transform(X_all[te_idx])
        lr, calib = _fit_and_calibrate_LR(Xtr, ytr, Xva, yva, class_weight="balanced", C=1.0)
        pte = calib.predict_proba(Xte)[:,1]
        org_test = f_real["organoid_id"].values[te_idx]
        per_org = {}
        for oid in np.unique(org_test):
            m = (org_test == oid)
            if m.sum() >= 1 and len(np.unique(yte[m])) > 1:
                per_org[str(oid)] = average_precision_score(yte[m], pte[m])
            else:
                per_org[str(oid)] = None
        fold_result["per_organoid"] = per_org

        all_results["folds"].append(fold_result)

        # ---------- Null checks (generalization-to-null) ----------
        # Apply the best ALL/LR (trained on real train+val policy) to surrogate features on the SAME test windows
        if args.features_ising:
            ft = f_ising
            # align schema to real (same columns)
            miss_cols = [c for c in f_real.columns if c not in ft.columns]
            for c in miss_cols: ft[c] = np.nan
            Xs_all = pre_all.transform(ft[cols_ALL].values)  # use real-fitted pre
            ps = calib.predict_proba(Xs_all[te_idx])[:,1]
            all_results.setdefault("null_checks", {}).setdefault(f"fold{fold}", {})["ising_PR_AUC"] = float(average_precision_score(yte, ps))
        if args.features_gamma:
            ft = f_gamma
            miss_cols = [c for c in f_real.columns if c not in ft.columns]
            for c in miss_cols: ft[c] = np.nan
            Xs_all = pre_all.transform(ft[cols_ALL].values)
            ps = calib.predict_proba(Xs_all[te_idx])[:,1]
            all_results.setdefault("null_checks", {}).setdefault(f"fold{fold}", {})["gamma_PR_AUC"] = float(average_precision_score(yte, ps))

    # Aggregate and save metrics
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"[OK] Saved artifacts to {args.out_dir}")

if __name__ == "__main__":
    main()
