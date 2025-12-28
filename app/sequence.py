from __future__ import annotations
import numpy as np


def build_sliding_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """
    (N, F) -> (N-seq_len+1, seq_len, F)
    Spójne z treningiem LSTM/meta.
    """
    n, f = X.shape
    if n < seq_len:
        return np.empty((0, seq_len, f), dtype=np.float32)

    try:
        from numpy.lib.stride_tricks import sliding_window_view
        Xw = sliding_window_view(X, window_shape=(seq_len, f))
        return Xw.squeeze(1).astype(np.float32, copy=False)
    except Exception:
        out = np.zeros((n - seq_len + 1, seq_len, f), dtype=np.float32)
        for i in range(n - seq_len + 1):
            out[i] = X[i : i + seq_len]
        return out
