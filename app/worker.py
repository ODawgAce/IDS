from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from app.detector import HybridDetector
from app.flowmeter import run_cicflowmeter, newest_csv
from app.preprocess import clean_flow_df, align_to_feature_cols


@dataclass
class WorkerConfig:
    dumpcap_path: str
    java_bin: str
    cicflow_jar: str
    jnetpcap_jar: str
    jnetpcap_dll_dir: str

    interface_index: int
    capture_seconds: int = 10

    # próg dla SCORE (OR RF+LSTM) – u Ciebie sensownie 0.25–0.40
    threshold: float = 0.30
    # ile flow >= threshold w jednym batchu, żeby zrobić alert
    min_hits: int = 5

    seq_len: int = 20

    runtime_dir: str = "runtime"
    # foldery “produkcyjne” (opcjonalne archiwum)
    pcap_dir: str = "runtime/pcaps"
    flows_dir: str = "runtime/flows"

    # izolowane foldery na pojedynczy batch (żeby CICFlow nie mielił starych pcapów)
    batch_pcap_dir: str = "runtime/_batch/pcaps"
    batch_flows_dir: str = "runtime/_batch/flows"


class IDSWorker(threading.Thread):
    def __init__(
        self,
        cfg: WorkerConfig,
        detector: HybridDetector,
        log_q: "queue.Queue[str]",
        alert_q: "queue.Queue[dict]",
    ):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.detector = detector
        self.log_q = log_q
        self.alert_q = alert_q
        self._stop_evt = threading.Event()

        self._ensure_dirs()

        # ile cech oczekuje pipeline (diagnostycznie)
        self.n_expected: Optional[int] = getattr(self.detector.preproc, "n_features_in_", None)

    def stop(self):
        self._stop_evt.set()

    def _log(self, msg: str):
        try:
            self.log_q.put_nowait(msg)
        except Exception:
            pass

    def _ensure_dirs(self):
        Path(self.cfg.runtime_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.pcap_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.flows_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.batch_pcap_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.batch_flows_dir).mkdir(parents=True, exist_ok=True)

    def _run_dumpcap(self, out_pcap: str) -> None:
        cmd = [
            self.cfg.dumpcap_path,
            "-i", str(self.cfg.interface_index),
            "-a", f"duration:{self.cfg.capture_seconds}",
            "-F", "pcap",
            "-w", out_pcap,
        ]
        self._log(f"Capturing {self.cfg.capture_seconds}s on iface {self.cfg.interface_index} ...")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    @staticmethod
    def _make_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
        n = X.shape[0]
        if n < seq_len:
            return np.empty((0, seq_len, X.shape[1]), dtype=np.float32)

        out = np.zeros((n - seq_len + 1, seq_len, X.shape[1]), dtype=np.float32)
        for i in range(n - seq_len + 1):
            out[i] = X[i: i + seq_len]
        return out

    def _load_features_for_runtime(self, csv_path: str, feature_cols: List[str]) -> pd.DataFrame:
        df = pd.read_csv(csv_path, low_memory=False, skipinitialspace=True)
        Xdf = clean_flow_df(df)
        Xdf = align_to_feature_cols(Xdf, feature_cols)
        return Xdf

    def _reset_batch_dirs(self):
        shutil.rmtree(self.cfg.batch_pcap_dir, ignore_errors=True)
        shutil.rmtree(self.cfg.batch_flows_dir, ignore_errors=True)
        Path(self.cfg.batch_pcap_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.batch_flows_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _stats_line(name: str, arr: Optional[np.ndarray]) -> str:
        if arr is None:
            return f"{name}: None"
        arr = np.asarray(arr)
        if arr.size == 0:
            return f"{name}: empty"
        q50, q90, q99 = np.quantile(arr, [0.5, 0.9, 0.99])
        return (
            f"{name}: max={float(arr.max()):.4f} mean={float(arr.mean()):.4f} "
            f"q50={float(q50):.4f} q90={float(q90):.4f} q99={float(q99):.4f}"
        )

    @staticmethod
    def _soft_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # soft-OR: 1 - (1-a)*(1-b)
        a = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)
        b = np.clip(np.asarray(b, dtype=np.float32), 0.0, 1.0)
        return 1.0 - (1.0 - a) * (1.0 - b)

    def run(self):
        if not getattr(self.detector, "feature_cols", None):
            self._log("ERROR: detector.feature_cols is missing (preproc.joblib must contain feature_cols).")
            return

        feature_cols = list(self.detector.feature_cols)
        pcap_counter = 0

        while not self._stop_evt.is_set():
            try:
                # 0) czysty batch (1 pcap -> 1 csv)
                self._reset_batch_dirs()

                # 1) capture do batchowego folderu
                pcap_counter += 1
                ts = time.strftime("%Y%m%d%H%M%S")
                pcap_name = f"capture_{pcap_counter:05d}_{ts}.pcap"
                pcap_path = str(Path(self.cfg.batch_pcap_dir) / pcap_name)

                self._run_dumpcap(pcap_path)

                # (opcjonalnie) archiwizacja pcap
                try:
                    shutil.copy2(pcap_path, str(Path(self.cfg.pcap_dir) / pcap_name))
                except Exception:
                    pass

                # 2) CICFlowMeter na batchu
                self._log("Generating flows (CICFlowMeter)...")
                run_cicflowmeter(
                    java_bin=self.cfg.java_bin,
                    cicflowmeter_jar=self.cfg.cicflow_jar,
                    jnetpcap_jar=self.cfg.jnetpcap_jar,
                    jnetpcap_dll_dir=self.cfg.jnetpcap_dll_dir,
                    pcap_dir=self.cfg.batch_pcap_dir,
                    out_dir=self.cfg.batch_flows_dir,
                )

                # 3) newest csv z batcha
                csv_path = str(newest_csv(self.cfg.batch_flows_dir))

                # (opcjonalnie) archiwizacja csv
                try:
                    shutil.copy2(csv_path, str(Path(self.cfg.flows_dir) / Path(csv_path).name))
                except Exception:
                    pass

                # 4) features identycznie jak trening
                Xdf = self._load_features_for_runtime(csv_path, feature_cols)
                if len(Xdf) == 0:
                    self._log(f"WARNING: Empty features after cleaning for {pcap_name}")
                    continue

                X_raw = Xdf.values.astype(np.float32, copy=False)
                X_scaled = self.detector.preproc.transform(X_raw).astype(np.float32, copy=False)

                # 5) sekwencje
                X_seq = self._make_sequences(X_scaled, self.cfg.seq_len)

                # 6) części + OUT
                rf_p, lstm_p, meta_p = self.detector.predict_parts(X_scaled, X_seq)
                out_p = self.detector.predict_proba(X_scaled, X_seq)

                # 7) SCORE = RF na początku + soft-OR(RF, LSTM) dla części sekwencyjnej
                score = rf_p.copy()
                if lstm_p is not None and np.asarray(lstm_p).size > 0 and rf_p.size >= self.cfg.seq_len:
                    rf_aligned = rf_p[self.cfg.seq_len - 1 :]
                    m = min(len(rf_aligned), len(lstm_p))
                    seq_score = self._soft_or(rf_aligned[:m], np.asarray(lstm_p)[:m])
                    score[self.cfg.seq_len - 1 : self.cfg.seq_len - 1 + m] = seq_score

                # 8) log statystyk
                self._log(self._stats_line("RF", rf_p))
                self._log(self._stats_line("LSTM", lstm_p))
                self._log(self._stats_line("META", meta_p))
                self._log(self._stats_line("OUT", out_p))
                self._log(self._stats_line("SCORE", score))

                if score.size == 0:
                    self._log("WARNING: Empty predictions")
                    continue

                thr = float(self.cfg.threshold)
                hits = int(np.sum(score >= thr))
                pmax = float(np.max(score))

                self._log(
                    f"[OK] {pcap_name}: rows={len(Xdf)} max_score={pmax:.4f} "
                    f"hits>thr={hits} thr={thr:.2f} K={int(self.cfg.min_hits)}"
                )

                if pmax >= thr and hits >= int(self.cfg.min_hits):
                    idx = int(np.argmax(score))
                    alert = {
                        "time": time.strftime("%H:%M:%S"),
                        "p_attack": pmax,
                        "source": pcap_name,
                        "details": f"max idx={idx}, hits={hits}, thr={thr}, K={self.cfg.min_hits}",
                    }
                    self.alert_q.put(alert)
                    self._log(f"ALERT: score={pmax:.4f} hits={hits} >= K={self.cfg.min_hits}")

            except subprocess.CalledProcessError as e:
                out = getattr(e, "stdout", None) or getattr(e, "output", None) or str(e)
                self._log(f"ERROR: subprocess failed:\n{out}")
            except Exception as e:
                self._log(f"ERROR: {type(e).__name__}: {e}")
