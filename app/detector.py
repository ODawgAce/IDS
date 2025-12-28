from __future__ import annotations

import joblib
import numpy as np
from tensorflow import keras


def _unwrap_meta(obj):
    if hasattr(obj, "predict_proba"):
        return obj
    if isinstance(obj, dict):
        for k in ("model", "meta", "clf", "estimator", "classifier"):
            if k in obj and hasattr(obj[k], "predict_proba"):
                return obj[k]
    raise TypeError("meta.joblib must contain an estimator with predict_proba (or dict wrapping it).")


def _unwrap_preproc(obj):
    if isinstance(obj, dict):
        preproc = obj.get("preproc", None)
        feature_cols = obj.get("feature_cols", None)
        if preproc is None:
            raise TypeError("preproc.joblib dict must contain key 'preproc'")
        return preproc, feature_cols
    return obj, None


class HybridDetector:
    def __init__(self, rf_path: str, lstm_path: str, meta_path: str, preproc_path: str, seq_len: int):
        self.rf = joblib.load(rf_path)
        self.lstm = keras.models.load_model(lstm_path)

        meta_obj = joblib.load(meta_path)
        self.meta = _unwrap_meta(meta_obj)

        pre_obj = joblib.load(preproc_path)
        self.preproc, self.feature_cols = _unwrap_preproc(pre_obj)

        self.seq_len = int(seq_len)

    def predict_parts(self, X_scaled: np.ndarray, X_seq_scaled: np.ndarray | None):
        """
        Zwraca:
          rf_p_full: (N,)
          lstm_p_seq: (N-seq_len+1,) lub None
          meta_p_seq: (N-seq_len+1,) lub None
        """
        rf_p_full = self.rf.predict_proba(X_scaled)[:, 1].astype(np.float32)

        if X_seq_scaled is None or len(X_seq_scaled) == 0 or len(rf_p_full) < self.seq_len:
            return rf_p_full, None, None

        lstm_p_seq = self.lstm.predict(X_seq_scaled, verbose=0).reshape(-1).astype(np.float32)

        rf_aligned = rf_p_full[self.seq_len - 1 :]
        m = min(len(rf_aligned), len(lstm_p_seq))
        rf_aligned = rf_aligned[:m]
        lstm_p_seq = lstm_p_seq[:m]

        Z = np.column_stack([rf_aligned, lstm_p_seq]).astype(np.float32)
        meta_p_seq = self.meta.predict_proba(Z)[:, 1].astype(np.float32)

        return rf_p_full, lstm_p_seq, meta_p_seq

    def predict_proba(self, X_scaled: np.ndarray, X_seq_scaled: np.ndarray | None) -> np.ndarray:
        rf_p_full, lstm_p_seq, meta_p_seq = self.predict_parts(X_scaled, X_seq_scaled)
        if meta_p_seq is None:
            return rf_p_full

        out = rf_p_full.copy()
        out[self.seq_len - 1 : self.seq_len - 1 + len(meta_p_seq)] = meta_p_seq
        return out
