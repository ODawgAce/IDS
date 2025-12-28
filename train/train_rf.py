from __future__ import annotations

# ---- MUST BE FIRST: make project root importable ----
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # .../IDS
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

try:
    from app.preprocess import clean_flow_df, make_binary_y, align_to_feature_cols
except ModuleNotFoundError:
    from preprocess import clean_flow_df, make_binary_y, align_to_feature_cols  # type: ignore


def _log(msg: str) -> None:
    print(msg, flush=True)


def load_xy(csv_path: Path, feature_cols: list[str], rows: int) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, nrows=rows, low_memory=False, skipinitialspace=True)
    if "Label" not in df.columns:
        raise ValueError(f"No Label column in {csv_path.name}")

    y = make_binary_y(df["Label"]).astype(np.int32, copy=False)
    Xdf = clean_flow_df(df)
    Xdf = align_to_feature_cols(Xdf, feature_cols)
    X = Xdf.values.astype(np.float32, copy=False)
    return X, y


def _sample_balanced(
    X: np.ndarray,
    y: np.ndarray,
    neg_pos_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep all positives; sample negatives to neg_pos_ratio * n_pos."""
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0:
        return X, y

    k_neg = int(min(len(neg), max(1, round(neg_pos_ratio * len(pos)))))
    neg_s = rng.choice(neg, size=k_neg, replace=False) if k_neg < len(neg) else neg
    idx = np.concatenate([pos, neg_s])
    rng.shuffle(idx)
    return X[idx], y[idx]


def _evaluate(name: str, model, X: np.ndarray, y: np.ndarray) -> None:
    pred = model.predict(X)
    try:
        proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba) if len(np.unique(y)) == 2 else float("nan")
        pr = average_precision_score(y, proba) if len(np.unique(y)) == 2 else float("nan")
    except Exception:
        auc, pr = float("nan"), float("nan")

    _log(f"\n=== {name} ===")
    _log(str(confusion_matrix(y, pred)))
    _log(classification_report(y, pred, digits=4))
    if not np.isnan(auc):
        _log(f"AUC={auc:.4f}  PR_AUC={pr:.4f}")


def main(
    data_dir: str,
    preproc_path: str,
    out_path: str,
    rows_per_file: int = 200000,
    n_estimators: int = 1200,
    max_depth: int = 30,
    min_samples_leaf: int = 5,
    min_samples_split: int = 10,
    max_features: str = "sqrt",
    use_oob: bool = True,
    seed: int = 42,
    friday_holdout: float = 0.20,   # % Friday rows -> valid
    neg_pos_ratio: float = 5.0,
    scale: bool = True,
) -> None:
    pre = joblib.load(preproc_path)
    if not isinstance(pre, dict) or "preproc" not in pre or "feature_cols" not in pre:
        raise ValueError("preproc.joblib must be a dict with keys: 'preproc' and 'feature_cols'")

    preproc = pre["preproc"]
    feature_cols = list(pre["feature_cols"])

    files = sorted(Path(data_dir).glob("*.csv"))
    if not files:
        raise RuntimeError(f"No CSV files found in: {data_dir}")

    friday_files = [f for f in files if "Friday" in f.name]
    other_files = [f for f in files if f not in friday_files]

    if friday_files:
        _log("[i] Friday files:")
        for f in friday_files:
            _log(f"  - {f.name}")
    else:
        _log("[w] No Friday files found by name; will do group split across all files.")

    rng = np.random.default_rng(seed)

    # --- Load all data ---
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    gs: list[np.ndarray] = []

    for gi, f in enumerate(files):
        X, y = load_xy(f, feature_cols, rows_per_file)
        Xs.append(X)
        ys.append(y)
        gs.append(np.full(len(y), gi, dtype=np.int32))
        _log(f"Loaded (rows={len(y)}, feats={X.shape[1]}) from {f.name} (attacks={int(y.sum())})")

    X_all = np.vstack(Xs)
    y_all = np.concatenate(ys)
    g_all = np.concatenate(gs)

    _log(f"[i] Total raw: X={X_all.shape} attacks={int(y_all.sum())} ({(y_all.mean()*100):.2f}%)")

    # --- Build train/valid split ---
    if friday_files:
        friday_gis = [i for i, f in enumerate(files) if f in friday_files]
        is_friday = np.isin(g_all, friday_gis)

        idx_fr = np.where(is_friday)[0]
        idx_oth = np.where(~is_friday)[0]

        if len(idx_fr) == 0:
            _log("[w] Friday samples empty after load; fallback to group split.")
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            tr_idx, va_idx = next(splitter.split(X_all, y_all, groups=g_all))
        else:
            # IMPORTANT: Friday is usually ONE group -> cannot GroupShuffleSplit it.
            # We split Friday by ROWS (stratified) to hold out a fraction.
            sss = StratifiedShuffleSplit(n_splits=1, test_size=friday_holdout, random_state=seed)
            fr_tr_rel, fr_va_rel = next(sss.split(np.zeros((len(idx_fr), 1)), y_all[idx_fr]))
            tr_idx = np.concatenate([idx_oth, idx_fr[fr_tr_rel]])
            va_idx = idx_fr[fr_va_rel]
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_idx, va_idx = next(splitter.split(X_all, y_all, groups=g_all))

    Xtr, ytr = X_all[tr_idx], y_all[tr_idx]
    Xva, yva = X_all[va_idx], y_all[va_idx]
    _log(f"[i] Split: train={Xtr.shape} (attacks={int(ytr.sum())})  valid={Xva.shape} (attacks={int(yva.sum())})")

    # --- Balance train ---
    Xtr_s, ytr_s = _sample_balanced(Xtr, ytr, neg_pos_ratio=neg_pos_ratio, rng=rng)
    _log(f"[i] After sampling: train={Xtr_s.shape} attacks={int(ytr_s.sum())} ({(ytr_s.mean()*100):.2f}%)")

    # --- Transform ---
    if scale:
        _log("[i] Applying preproc.transform()")
        Xtr_s = preproc.transform(Xtr_s)
        Xva_t = preproc.transform(Xva)
    else:
        _log("[i] Training RF on RAW features (no scaling)")
        Xva_t = Xva

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
        n_jobs=-1,
        random_state=seed,
        class_weight="balanced_subsample",
        bootstrap=True,
        oob_score=bool(use_oob),
    )

    _log("[*] Fitting RandomForest...")
    rf.fit(Xtr_s, ytr_s)

    if use_oob:
        try:
            _log(f"[i] OOB score: {rf.oob_score_:.4f}")
        except Exception:
            pass

    _evaluate("VALID", rf, Xva_t, yva)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, out_path)
    _log(f"[✓] Saved RF to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--preproc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows_per_file", type=int, default=200000)

    ap.add_argument("--n_estimators", type=int, default=1200)
    ap.add_argument("--max_depth", type=int, default=30)
    ap.add_argument("--min_samples_leaf", type=int, default=5)
    ap.add_argument("--min_samples_split", type=int, default=10)
    ap.add_argument("--max_features", type=str, default="sqrt")
    ap.add_argument("--oob", action="store_true")

    ap.add_argument("--friday_holdout", type=float, default=0.20)
    ap.add_argument("--neg_pos_ratio", type=float, default=5.0)
    ap.add_argument("--no_scale", action="store_true")

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    main(
        data_dir=args.data_dir,
        preproc_path=args.preproc,
        out_path=args.out,
        rows_per_file=args.rows_per_file,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        max_features=args.max_features,
        use_oob=args.oob,
        friday_holdout=args.friday_holdout,
        neg_pos_ratio=args.neg_pos_ratio,
        scale=(not args.no_scale),
        seed=args.seed,
    )
