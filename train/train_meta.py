from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # katalog projektu (IDS)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import argparse
from collections import deque
from typing import Iterable, Tuple, List

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from app.preprocess import clean_flow_df, make_binary_y, align_to_feature_cols


def _log(msg: str) -> None:
    print(msg, flush=True)


def _list_files(data_dir: str) -> List[Path]:
    files = sorted(Path(data_dir).glob("*.csv"))
    if not files:
        raise RuntimeError(f"No CSV files found in: {data_dir}")
    return files


def _split_train_valid(files: List[Path], friday_valid: bool) -> Tuple[List[Path], List[Path]]:
    if not friday_valid:
        return files, []

    train_files = [f for f in files if "friday" not in f.name.lower()]
    valid_files = [f for f in files if "friday" in f.name.lower()]

    if not valid_files:
        raise RuntimeError("Requested --friday_valid, but no Friday file found in data_dir.")

    return train_files, valid_files


def _prepare_Xy_chunk(
    df: pd.DataFrame,
    feature_cols: list[str],
    preproc,
) -> Tuple[np.ndarray, np.ndarray]:
    if "Label" not in df.columns:
        raise ValueError("Chunk missing Label column")

    y_full = make_binary_y(df["Label"]).astype(np.int32, copy=False)

    Xdf = clean_flow_df(df)
    Xdf = align_to_feature_cols(Xdf, feature_cols)

    X = Xdf.values.astype(np.float32, copy=False)
    X = preproc.transform(X).astype(np.float32, copy=False)
    return X, y_full


def _balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    """
    Zwraca sample_weight w stylu 'balanced' dla batcha.
    Działa nawet jeśli batch ma tylko jedną klasę.
    """
    y = y.astype(np.int32, copy=False)
    n = len(y)
    if n == 0:
        return np.array([], dtype=np.float32)

    n1 = int(y.sum())
    n0 = n - n1

    # jeśli batch ma jedną klasę -> wagi wszystkie = 1
    if n0 == 0 or n1 == 0:
        return np.ones(n, dtype=np.float32)

    # "balanced": waga klasy k ~ N / (2 * Nk)
    w0 = n / (2.0 * n0)
    w1 = n / (2.0 * n1)
    return np.where(y == 1, w1, w0).astype(np.float32)


def _meta_stream_for_file(
    csv_path: Path,
    feature_cols: list[str],
    preproc,
    rf,
    lstm,
    seq_len: int,
    *,
    rows_per_file: int,
    chunksize: int,
    batch_sequences: int,
    meta_sample: int = 0,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Generator zwracający paczki (Xm_batch, y_batch) dla meta:
      Xm_batch: (B, 2) -> [p_rf_aligned, p_lstm]
      y_batch:  (B,)

    Streaming chunkami + rolling buffer seq_len-1.
    """
    buf_X = deque(maxlen=seq_len - 1)
    buf_y = deque(maxlen=seq_len - 1)

    rng = np.random.default_rng(42)
    n_seen = 0

    reader = pd.read_csv(csv_path, low_memory=False, skipinitialspace=True, chunksize=chunksize)
    for chunk in reader:
        if rows_per_file and n_seen >= rows_per_file:
            break

        if rows_per_file:
            remaining = rows_per_file - n_seen
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()

        n_seen += len(chunk)

        X, y = _prepare_Xy_chunk(chunk, feature_cols, preproc)

        if seq_len > 1 and len(buf_X) > 0:
            X_prev = np.stack(list(buf_X), axis=0).astype(np.float32, copy=False)
            y_prev = np.array(list(buf_y), dtype=np.int32)
            X_all = np.vstack([X_prev, X]).astype(np.float32, copy=False)
            y_all = np.concatenate([y_prev, y]).astype(np.int32, copy=False)
        else:
            X_all = X
            y_all = y

        if seq_len > 1:
            tail = X_all[-(seq_len - 1):] if len(X_all) >= (seq_len - 1) else X_all
            tail_y = y_all[-(seq_len - 1):] if len(y_all) >= (seq_len - 1) else y_all
            buf_X.clear()
            buf_y.clear()
            for i in range(len(tail)):
                buf_X.append(tail[i])
                buf_y.append(int(tail_y[i]))

        if len(X_all) < seq_len:
            continue

        p_rf = rf.predict_proba(X_all)[:, 1].astype(np.float32, copy=False)

        start = seq_len - 1
        end = len(X_all)

        idx = start
        while idx < end:
            j = min(idx + batch_sequences, end)
            ends = np.arange(idx, j, dtype=np.int32)
            B = len(ends)

            Xseq = np.empty((B, seq_len, X_all.shape[1]), dtype=np.float32)
            for bi, e in enumerate(ends):
                Xseq[bi] = X_all[e - seq_len + 1 : e + 1]

            p_lstm = lstm.predict(Xseq, verbose=0).reshape(-1).astype(np.float32, copy=False)
            p_rf_aligned = p_rf[ends].astype(np.float32, copy=False)
            y_aligned = y_all[ends].astype(np.int32, copy=False)

            Xm = np.column_stack([p_rf_aligned, p_lstm]).astype(np.float32, copy=False)

            if meta_sample and meta_sample > 0 and len(Xm) > meta_sample:
                take = rng.choice(len(Xm), size=meta_sample, replace=False)
                Xm = Xm[take]
                y_aligned = y_aligned[take]

            yield Xm, y_aligned
            idx = j


def _eval_meta_on_files(
    meta,
    valid_files: List[Path],
    feature_cols: list[str],
    preproc,
    rf,
    lstm,
    seq_len: int,
    *,
    rows_per_file: int,
    chunksize: int,
    batch_sequences: int,
) -> None:
    if not valid_files:
        _log("[i] No validation files -> skipping evaluation.")
        return

    _log("[*] Evaluating meta on VALID files...")
    y_true_all = []
    y_hat_all = []

    for f in valid_files:
        _log(f"  - {f.name}")
        for Xm, yb in _meta_stream_for_file(
            f, feature_cols, preproc, rf, lstm, seq_len,
            rows_per_file=rows_per_file,
            chunksize=chunksize,
            batch_sequences=batch_sequences,
            meta_sample=0,
        ):
            # SGDClassifier(log_loss) ma predict_proba
            p = meta.predict_proba(Xm)[:, 1]
            yhat = (p >= 0.5).astype(np.int32)
            y_true_all.append(yb)
            y_hat_all.append(yhat)

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.int32)
    y_hat = np.concatenate(y_hat_all) if y_hat_all else np.array([], dtype=np.int32)

    if len(y_true) == 0:
        _log("[!] No validation samples produced.")
        return

    _log("Confusion matrix (meta, thr=0.5):")
    _log(str(confusion_matrix(y_true, y_hat)))
    _log(classification_report(y_true, y_hat, digits=4))


def main(
    data_dir: str,
    preproc_path: str,
    rf_path: str,
    lstm_path: str,
    out_path: str,
    *,
    seq_len: int,
    rows_per_file: int = 800000,
    chunksize: int = 200000,
    batch_sequences: int = 512,
    friday_valid: bool = True,
    meta_sample: int = 0,
) -> None:
    pre = joblib.load(preproc_path)
    if not isinstance(pre, dict) or "preproc" not in pre or "feature_cols" not in pre:
        raise ValueError("preproc.joblib must be a dict with keys: 'preproc' and 'feature_cols'")

    preproc = pre["preproc"]
    feature_cols = list(pre["feature_cols"])

    _log("[*] Loading base models...")
    rf = joblib.load(rf_path)
    lstm = keras.models.load_model(lstm_path)

    files = _list_files(data_dir)
    train_files, valid_files = _split_train_valid(files, friday_valid=friday_valid)

    _log("[i] Train files:")
    for f in train_files:
        _log(f"  - {f.name}")
    if valid_files:
        _log("[i] Valid files:")
        for f in valid_files:
            _log(f"  - {f.name}")

    # UWAGA: class_weight nie może być 'balanced' przy partial_fit -> robimy sample_weight ręcznie
    meta = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        learning_rate="optimal",
        max_iter=1,
        tol=None,
        class_weight=None,
        random_state=42,
    )

    classes = np.array([0, 1], dtype=np.int32)
    fitted = False

    _log("[*] Training meta (streaming SGDClassifier + manual balanced sample_weight)...")
    for f in train_files:
        _log(f"[*] Streaming train file: {f.name}")
        for Xm, yb in _meta_stream_for_file(
            f, feature_cols, preproc, rf, lstm, seq_len,
            rows_per_file=rows_per_file,
            chunksize=chunksize,
            batch_sequences=batch_sequences,
            meta_sample=meta_sample,
        ):
            sw = _balanced_sample_weight(yb)

            if not fitted:
                meta.partial_fit(Xm, yb, classes=classes, sample_weight=sw)
                fitted = True
            else:
                meta.partial_fit(Xm, yb, sample_weight=sw)

    if not fitted:
        raise RuntimeError("Meta training produced 0 samples. Check seq_len / data / preprocessing.")

    _log("[✓] Meta training done.")

    _eval_meta_on_files(
        meta,
        valid_files,
        feature_cols,
        preproc,
        rf,
        lstm,
        seq_len,
        rows_per_file=rows_per_file,
        chunksize=chunksize,
        batch_sequences=batch_sequences,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "meta": meta,
            "base": {"rf": os.path.basename(rf_path), "lstm": os.path.basename(lstm_path)},
            "seq_len": seq_len,
            "features": ["p_rf", "p_lstm"],
        },
        out_path,
    )
    _log(f"[✓] Saved meta-model bundle to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--preproc", required=True)
    ap.add_argument("--rf", required=True)
    ap.add_argument("--lstm", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--rows_per_file", type=int, default=800000)
    ap.add_argument("--chunksize", type=int, default=200000)
    ap.add_argument("--batch_sequences", type=int, default=512)

    ap.add_argument("--friday_valid", action="store_true", help="Use Friday as validation (train on other days)")
    ap.add_argument("--meta_sample", type=int, default=0, help="Optional per-batch subsample (e.g. 2048)")

    args = ap.parse_args()

    main(
        data_dir=args.data_dir,
        preproc_path=args.preproc,
        rf_path=args.rf,
        lstm_path=args.lstm,
        out_path=args.out,
        seq_len=args.seq_len,
        rows_per_file=args.rows_per_file,
        chunksize=args.chunksize,
        batch_sequences=args.batch_sequences,
        friday_valid=args.friday_valid,
        meta_sample=args.meta_sample,
    )
