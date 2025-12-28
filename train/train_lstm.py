from __future__ import annotations

import os
import glob
import argparse
from typing import List, Tuple, Iterator, Optional

import numpy as np
import pandas as pd
import joblib

import tensorflow as tf
from tensorflow import keras
from keras import layers


# =========================
# Helpers
# =========================

def _log(msg: str) -> None:
    print(msg, flush=True)


def _cleanup_tmp_files(data_dir: str) -> None:
    for p in glob.glob(os.path.join(data_dir, "*.tmp_head.csv")):
        try:
            os.remove(p)
        except Exception:
            pass


def _list_csv_files(data_dir: str) -> List[str]:
    _cleanup_tmp_files(data_dir)
    files = sorted(
        p for p in glob.glob(os.path.join(data_dir, "*.csv"))
        if not p.lower().endswith(".tmp_head.csv")
    )
    return files


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]

    # czasem pierwszy wiersz bywa "nagłówkiem w danych"
    if len(df) > 0:
        first_row = df.iloc[0].astype(str).str.strip().values
        if any(v in set(df.columns) for v in first_row):
            df = df.iloc[1:].copy()
    return df


def _label_to_binary(y: pd.Series) -> np.ndarray:
    y_str = y.astype(str).str.strip().str.upper()
    return (y_str != "BENIGN").astype(np.int32).to_numpy()


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


_TS_FORMATS = [
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %I:%M:%S %p",
    "%m/%d/%Y %I:%M %p",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
]


def _try_parse_ts(ts_series: pd.Series) -> pd.Series:
    s = ts_series.astype(str).str.strip()
    out = None
    for fmt in _TS_FORMATS:
        tmp = pd.to_datetime(s, format=fmt, errors="coerce")
        out = tmp if out is None else out.fillna(tmp)
        if out.notna().mean() > 0.98:
            break
    if out is None or out.notna().sum() == 0:
        out = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return out


def _make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
    *,
    label_frac: float = 0.20,
    drop_mixed: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding window:
      X_seq: (N-seq_len+1, seq_len, F)

    Label okna:
      y_seq = 1 jeśli mean(y_window) >= label_frac
      drop_mixed=True usuwa okna mieszane (stabilizuje, ale zmienia rozkład)
    """
    n, f = X.shape
    if n < seq_len:
        return (
            np.empty((0, seq_len, f), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    # szybkie okna bez pętli
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        Xw = sliding_window_view(X, window_shape=(seq_len, f))
        Xs = Xw.squeeze(1).astype(np.float32, copy=False)
    except Exception:
        Xs = np.zeros((n - seq_len + 1, seq_len, f), dtype=np.float32)
        for i in range(n - seq_len + 1):
            Xs[i] = X[i : i + seq_len]

    y = y.astype(np.int32, copy=False)
    w_sum = np.convolve(y, np.ones(seq_len, dtype=np.int32), mode="valid")
    w_frac = w_sum / float(seq_len)
    ys = (w_frac >= label_frac).astype(np.int32)

    if drop_mixed:
        keep = (w_frac <= 0.05) | (w_frac >= label_frac)
        Xs = Xs[keep]
        ys = ys[keep]

    return Xs, ys


def _extract_feature_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    if "Label" not in df.columns:
        raise ValueError("missing Label column")

    # sort (ważne dla sekwencji)
    if "Timestamp" in df.columns:
        ts = _try_parse_ts(df["Timestamp"])
        if ts.notna().any():
            df = df.assign(_ts=ts).sort_values("_ts").drop(columns=["_ts"])

    y = _label_to_binary(df["Label"])

    drop_cols = [
        "Label", "Flow ID", "Timestamp",
        "Src IP", "Dst IP", "Source IP", "Destination IP",
        "Src Port", "Dst Port", "Protocol",
    ]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    df.columns = [str(c).strip() for c in df.columns]

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    df = df[feature_cols]
    df = _coerce_numeric(df)

    X = df.values.astype(np.float32, copy=False)
    return X, y


# =========================
# Model
# =========================

def build_lstm(seq_len: int, n_features: int, lr: float = 1e-4) -> keras.Model:
    inp = keras.Input(shape=(seq_len, n_features), name="x")

    x = layers.LayerNormalization()(inp)

    x = layers.Bidirectional(
        layers.LSTM(
            32,
            return_sequences=True,
            dropout=0.25,
            recurrent_dropout=0.0,
            kernel_regularizer=keras.regularizers.l2(1e-5),
        )
    )(x)

    x = layers.Bidirectional(
        layers.LSTM(
            16,
            return_sequences=False,
            dropout=0.25,
            recurrent_dropout=0.0,
            kernel_regularizer=keras.regularizers.l2(1e-5),
        )
    )(x)

    x = layers.Dense(16, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-5))(x)
    x = layers.Dropout(0.50)(x)

    out = layers.Dense(1, activation="sigmoid", name="p")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="AUC", curve="ROC"),
            keras.metrics.AUC(name="PR_AUC", curve="PR"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# =========================
# Streaming sequences -> tf.data
# =========================

def _iter_sequences_from_files(
    files: List[str],
    feature_cols: List[str],
    preproc,
    *,
    seq_len: int,
    rows_per_file: int,
    chunksize: int,
    label_frac: float,
    drop_mixed: bool,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generator: yield (X_seq, y_seq) batches as individual sequences (not batched).
    """
    while True:
        for csv_path in files:
            base = os.path.basename(csv_path)
            _log(f"[i] streaming file: {base}")

            rows_seen = 0

            for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
                chunk = _normalize_columns(chunk)
                if "Label" not in chunk.columns:
                    raise ValueError(f"{base} missing Label column")

                # limit rows per file (przydatne jak chcesz skrócić trening)
                if rows_per_file > 0 and (rows_seen + len(chunk)) > rows_per_file:
                    chunk = chunk.iloc[: max(0, rows_per_file - rows_seen)]
                rows_seen += len(chunk)
                if len(chunk) == 0:
                    break

                X_rows, y_rows = _extract_feature_matrix(chunk, feature_cols)

                # skalowanie na poziomie wierszy (preproc jest fit na wierszach)
                Xs = preproc.transform(X_rows).astype(np.float32, copy=False)

                X_seq, y_seq = _make_sequences(
                    Xs, y_rows, seq_len,
                    label_frac=label_frac,
                    drop_mixed=drop_mixed,
                )
                if X_seq.shape[0] == 0:
                    continue

                # yield pojedyncze sekwencje (dataset zrobi batching)
                for i in range(X_seq.shape[0]):
                    yield X_seq[i], y_seq[i]

                if rows_per_file > 0 and rows_seen >= rows_per_file:
                    break

            _log(f"[i] done: {base} rows_seen={rows_seen}")


def _make_dataset(
    files: List[str],
    feature_cols: List[str],
    preproc,
    *,
    seq_len: int,
    rows_per_file: int,
    chunksize: int,
    label_frac: float,
    drop_mixed: bool,
    batch_sequences: int,
    shuffle_buffer: int,
) -> tf.data.Dataset:
    output_signature = (
        tf.TensorSpec(shape=(seq_len, len(feature_cols)), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    gen = lambda: _iter_sequences_from_files(
        files,
        feature_cols,
        preproc,
        seq_len=seq_len,
        rows_per_file=rows_per_file,
        chunksize=chunksize,
        label_frac=label_frac,
        drop_mixed=drop_mixed,
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    if shuffle_buffer and shuffle_buffer > 0:
        ds = ds.shuffle(buffer_size=int(shuffle_buffer), reshuffle_each_iteration=True)

    ds = ds.batch(int(batch_sequences), drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# =========================
# Main
# =========================

def main(
    data_dir: str,
    preproc_path: str,
    out_path: str,
    *,
    seq_len: int,
    rows_per_file: int,
    chunksize: int,
    batch_sequences: int,
    epochs: int,
    steps_per_epoch: int,
    validation_steps: int,
    label_frac: float,
    drop_mixed: bool,
    friday_valid: bool,
    train_shuffle_buffer: int,
    val_shuffle_buffer: int,
) -> None:
    pre = joblib.load(preproc_path)
    if not isinstance(pre, dict) or "preproc" not in pre or "feature_cols" not in pre:
        raise ValueError("preproc.joblib must be a dict with keys: 'preproc' and 'feature_cols'")
    preproc = pre["preproc"]
    feature_cols = list(pre["feature_cols"])

    csv_files = _list_csv_files(data_dir)
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {data_dir}")

    train_files = csv_files
    valid_files: List[str] = []

    if friday_valid:
        train_files = [f for f in csv_files if "friday" not in os.path.basename(f).lower()]
        valid_files = [f for f in csv_files if "friday" in os.path.basename(f).lower()]
        if not valid_files:
            raise RuntimeError("friday_valid=True but no Friday file found in data_dir.")

    _log("[*] Training LSTM (streaming, low RAM)...")
    if friday_valid:
        _log("[i] Train files:")
        for f in train_files:
            _log(f"  - {os.path.basename(f)}")
        _log("[i] Valid files:")
        for f in valid_files:
            _log(f"  - {os.path.basename(f)}")
    else:
        _log("[i] Using ALL files for train+val sampling (not day-held-out).")

    model = build_lstm(seq_len=seq_len, n_features=len(feature_cols), lr=1e-4)
    model.summary()

    ds_tr = _make_dataset(
        train_files,
        feature_cols,
        preproc,
        seq_len=seq_len,
        rows_per_file=rows_per_file,
        chunksize=chunksize,
        label_frac=label_frac,
        drop_mixed=drop_mixed,
        batch_sequences=batch_sequences,
        shuffle_buffer=train_shuffle_buffer,
    ).repeat()

    if friday_valid:
        ds_va = _make_dataset(
            valid_files,
            feature_cols,
            preproc,
            seq_len=seq_len,
            rows_per_file=rows_per_file,
            chunksize=chunksize,
            label_frac=label_frac,
            drop_mixed=drop_mixed,
            batch_sequences=batch_sequences,
            shuffle_buffer=val_shuffle_buffer,   # <<< KLUCZOWE: tasuje Friday
        ).repeat()
    else:
        ds_va = None

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_PR_AUC" if friday_valid else "PR_AUC",
            mode="max",
            patience=3,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_PR_AUC" if friday_valid else "PR_AUC",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=out_path + ".best.keras",
            monitor="val_PR_AUC" if friday_valid else "PR_AUC",
            mode="max",
            save_best_only=True,
        ),
    ]

    if friday_valid:
        history = model.fit(
            ds_tr,
            epochs=int(epochs),
            steps_per_epoch=int(steps_per_epoch),
            validation_data=ds_va,
            validation_steps=int(validation_steps),
            callbacks=callbacks,
            verbose=1,
        )
    else:
        history = model.fit(
            ds_tr,
            epochs=int(epochs),
            steps_per_epoch=int(steps_per_epoch),
            callbacks=callbacks,
            verbose=1,
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model.save(out_path)
    _log(f"[✓] Saved LSTM to {out_path}")
    _log(f"[i] Also saved best checkpoint to {out_path}.best.keras")

    if friday_valid:
        _log("[i] TIP: jeśli dalej val_recall niski, spróbuj:")
        _log("  - zwiększyć --validation_steps (np. 2000)")
        _log("  - zwiększyć --val_shuffle_buffer (np. 100000)")
        _log("  - zmniejszyć --label_frac (np. 0.10-0.15) żeby łatwiej oznaczać okna jako atak")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--preproc", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--rows_per_file", type=int, default=800000)
    ap.add_argument("--chunksize", type=int, default=200000)
    ap.add_argument("--batch_sequences", type=int, default=512)

    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps_per_epoch", type=int, default=2000)
    ap.add_argument("--validation_steps", type=int, default=1200)

    ap.add_argument("--label_frac", type=float, default=0.20)
    ap.add_argument("--drop_mixed", action="store_true", help="Drop mixed windows (more stable but changes distribution)")

    ap.add_argument("--friday_valid", action="store_true", help="Use Friday file(s) only as validation")

    ap.add_argument("--train_shuffle_buffer", type=int, default=20000)
    ap.add_argument("--val_shuffle_buffer", type=int, default=50000)

    args = ap.parse_args()

    main(
        data_dir=args.data_dir,
        preproc_path=args.preproc,
        out_path=args.out,
        seq_len=args.seq_len,
        rows_per_file=args.rows_per_file,
        chunksize=args.chunksize,
        batch_sequences=args.batch_sequences,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        label_frac=args.label_frac,
        drop_mixed=args.drop_mixed,
        friday_valid=args.friday_valid,
        train_shuffle_buffer=args.train_shuffle_buffer,
        val_shuffle_buffer=args.val_shuffle_buffer,
    )
