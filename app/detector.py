from __future__ import annotations


import logging

import numpy as np

import joblib                  # <--- PRZENIESIONE NA GÓRĘ

from tensorflow import keras   # <--- PRZENIESIONE NA GÓRĘ


print("### NEW DETECTOR LOADED ###")


logger = logging.getLogger("IDS")


# Funkcja _unwrap_preproc usunięta, logika przeniesiona do klasy


class HybridDetector:

    """

    Wersja bez META:

      - RF + LSTM

      - predict_parts -> (rf_p_full, lstm_p_seq, None)

      - predict_proba -> tylko RF

        (worker liczy SCORE jako soft-OR(RF, LSTM))

    """

    def __init__(

        self,

        rf_path: str,

        lstm_path: str,

        preproc_path: str,

        seq_len: int,

    ):

        try:

            # Wczytywanie modeli

            self.rf = joblib.load(rf_path)

            self.lstm = keras.models.load_model(lstm_path)

            # --- LOGIKA UNWRAP (zamiast funkcji zewnętrznej) ---

            # Zmieniamy nazwę zmiennej na 'loaded_data' dla bezpieczeństwa

            loaded_data = joblib.load(preproc_path)

            if isinstance(loaded_data, dict):

                self.preproc = loaded_data.get("preproc")

                self.feature_cols = loaded_data.get("feature_cols")

                if self.preproc is None:

                    raise TypeError(

                        "preproc.joblib dict must contain key 'preproc'")

            else:

                self.preproc = loaded_data

                self.feature_cols = None

            # ---------------------------------------------------

            self.seq_len = int(seq_len)

            logger.info("[OK] HybridDetector initialized")

        except Exception as e:

            logger.warning(f"[HybridDetector init failed] {e}")

            raise  # GUI ma to zobaczyć

    def predict_parts(

        self,

        X_scaled: np.ndarray,

        X_seq_scaled: np.ndarray | None,

    ):
        """

        Zwraca:

          rf_p_full: (N,)

          lstm_p_seq: (N - seq_len + 1,) lub None

          meta_p_seq: zawsze None

        """

        try:

            rf_p_full = self.rf.predict_proba(

                X_scaled)[:, 1].astype(np.float32)

            if (

                X_seq_scaled is None

                or len(X_seq_scaled) == 0

                or len(rf_p_full) < self.seq_len

            ):

                return rf_p_full, None, None

            lstm_p_seq = (

                self.lstm.predict(X_seq_scaled, verbose=0)

                .reshape(-1)

                .astype(np.float32)

            )

            return rf_p_full, lstm_p_seq, None

        except Exception as e:

            logger.warning(f"[predict_parts failed] {e}")

            return (

                np.zeros(len(X_scaled), dtype=np.float32),

                None,

                None,

            )

    # def predict_proba(

    #     self,

    #     X_scaled: np.ndarray,

    #     # Ten argument jest tu właściwie zbędny, ale zostawiamy dla zgodności interfejsu

    #     X_seq_scaled: np.ndarray | None = None,

    # ) -> np.ndarray:

    #     # ZMIANA: Zamiast wywoływać predict_parts (które liczy LSTM),

    #     # wywołujemy bezpośrednio lekki model RF.

    #     # Zakładamy, że interesuje nas klasa 1 (atak)

    #     return self.rf.predict_proba(X_scaled)[:, 1].astype(np.float32)
