from __future__ import annotations

import joblib
import numpy as np
from tensorflow import keras


def _unwrap_preproc(obj):
    if isinstance(obj, dict):
        preproc = obj.get("preproc", None)
        feature_cols = obj.get("feature_cols", None)
        if preproc is None:
            raise TypeError("preproc.joblib dict must contain key 'preproc'")
        return preproc, feature_cols
    return obj, None


class HybridDetector:
    """
    Wersja bez META:
      - RF + LSTM
      - predict_parts zwraca (rf_p_full, lstm_p_seq, None)
      - predict_proba zwraca RF (żeby nie mieszać logiki workerowi),
        a worker i tak liczy SCORE soft-OR(RF, LSTM).
    """

    def __init__(self, rf_path: str, lstm_path: str, preproc_path: str, seq_len: int):
        self.rf = joblib.load(rf_path)
        self.lstm = keras.models.load_model(lstm_path)

        pre_obj = joblib.load(preproc_path)
        self.preproc, self.feature_cols = _unwrap_preproc(pre_obj)

        self.seq_len = int(seq_len)

    def predict_parts(self, X_scaled: np.ndarray, X_seq_scaled: np.ndarray | None):
        """
        Zwraca:
          rf_p_full: (N,)
          lstm_p_seq: (N-seq_len+1,) lub None
          meta_p_seq: zawsze None (meta usunięta)
        """
        rf_p_full = self.rf.predict_proba(X_scaled)[:, 1].astype(np.float32)

        if X_seq_scaled is None or len(X_seq_scaled) == 0 or len(rf_p_full) < self.seq_len:
            return rf_p_full, None, None

        lstm_p_seq = self.lstm.predict(X_seq_scaled, verbose=0).reshape(-1).astype(np.float32)
        return rf_p_full, lstm_p_seq, None

    def predict_proba(self, X_scaled: np.ndarray, X_seq_scaled: np.ndarray | None) -> np.ndarray:
        """
        Zostawiamy RF jako "OUT", bo docelowy alarm i tak liczysz w workerze jako SCORE (soft-OR).
        """
        rf_p_full, _lstm_p_seq, _ = self.predict_parts(X_scaled, X_seq_scaled)
        return rf_p_full
