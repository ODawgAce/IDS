from __future__ import annotations

import os
import queue
import tkinter as tk
from tkinter import ttk
from dataclasses import asdict

from app.worker import IDSWorker, WorkerConfig
from app.detector import HybridDetector


class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IDS (RF + LSTM + Meta)")
        self.geometry("1100x650")

        self.detector: HybridDetector | None = None
        self.worker: IDSWorker | None = None

        self.log_q: "queue.Queue[str]" = queue.Queue()
        self.alert_q: "queue.Queue[dict]" = queue.Queue()

        self.batch_count = 0
        self.alert_count = 0

        self._build_ui()
        self._load_models()
        self.after(200, self._poll_queues)

    def _build_ui(self):
        frm = ttk.LabelFrame(self, text="Sterowanie")
        frm.pack(fill="x", padx=10, pady=10)

        row = ttk.Frame(frm)
        row.pack(fill="x", padx=10, pady=8)

        ttk.Label(row, text="Interfejs (dumpcap -i):").pack(side="left")
        self.iface_var = tk.StringVar(value="1")
        self.iface_box = ttk.Combobox(row, textvariable=self.iface_var, width=25, values=["1", "2", "3", "4"])
        self.iface_box.pack(side="left", padx=8)
        ttk.Label(row, text="(np. 1 = Wi-Fi)").pack(side="left")

        row2 = ttk.Frame(frm)
        row2.pack(fill="x", padx=10, pady=8)

        ttk.Label(row2, text="Próg alertu:").pack(side="left")
        self.thr_var = tk.DoubleVar(value=0.50)
        self.thr_spin = ttk.Spinbox(row2, from_=0.0, to=1.0, increment=0.01, textvariable=self.thr_var, width=8)
        self.thr_spin.pack(side="left", padx=8)

        ttk.Label(row2, text="Czas batcha (s):").pack(side="left", padx=(20, 0))
        self.cap_var = tk.IntVar(value=60)
        self.cap_spin = ttk.Spinbox(row2, from_=5, to=180, increment=5, textvariable=self.cap_var, width=6)
        self.cap_spin.pack(side="left", padx=8)

        row3 = ttk.Frame(frm)
        row3.pack(fill="x", padx=10, pady=(0, 10))

        self.start_btn = ttk.Button(row3, text="Start", command=self.on_start)
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.stop_btn = ttk.Button(row3, text="Stop", command=self.on_stop, state="disabled")
        self.stop_btn.pack(side="left", fill="x", expand=True, padx=(5, 0))

        stat = ttk.Frame(self)
        stat.pack(fill="x", padx=10, pady=(0, 10))

        self.status_var = tk.StringVar(value="Status: idle")
        ttk.Label(stat, textvariable=self.status_var).pack(side="left")

        self.counters_var = tk.StringVar(value="Batches: 0 | Alerts: 0")
        ttk.Label(stat, textvariable=self.counters_var).pack(side="right")

        log_frm = ttk.LabelFrame(self, text="Log")
        log_frm.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.log_text = tk.Text(log_frm, height=12)
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.log_text.configure(state="disabled")

        alert_frm = ttk.LabelFrame(self, text="Alerty")
        alert_frm.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        cols = ("time", "p_attack", "source", "details")
        self.tree = ttk.Treeview(alert_frm, columns=cols, show="headings")
        self.tree.heading("time", text="Czas")
        self.tree.heading("p_attack", text="P(attack)")
        self.tree.heading("source", text="Źródło")
        self.tree.heading("details", text="Szczegóły")

        self.tree.column("time", width=90, anchor="w")
        self.tree.column("p_attack", width=90, anchor="w")
        self.tree.column("source", width=320, anchor="w")
        self.tree.column("details", width=520, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)

    def _append_log(self, line: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _load_models(self):
        preproc = os.path.join("artifacts", "preproc.joblib")
        rf = os.path.join("artifacts", "rf.joblib")
        lstm = os.path.join("artifacts", "lstm.keras.best.keras")
        meta = os.path.join("artifacts", "meta.joblib")

        self.detector = HybridDetector(
            rf_path=rf,
            lstm_path=lstm,
            meta_path=meta,
            preproc_path=preproc,
            seq_len=20,
        )
        self._append_log("[OK] Models loaded.")
        self.status_var.set("Status: ready")

    def _make_config(self) -> WorkerConfig:
        return WorkerConfig(
            dumpcap_path=r"C:\Program Files\Wireshark\dumpcap.exe",
            java_bin="java",
            cicflow_jar=r"C:\tools\CICFlowMeter\target\CICFlowMeterV3-0.0.4-SNAPSHOT.jar",
            jnetpcap_jar=r"C:\tools\jnetpcap-1.4.r1425\jnetpcap.jar",
            jnetpcap_dll_dir=r"C:\tools\jnetpcap-1.4.r1425",
            interface_index=int(self.iface_var.get()),
            capture_seconds=int(self.cap_var.get()),
            threshold=float(self.thr_var.get()),
            seq_len=20,
            runtime_dir="runtime",
            pcap_dir="runtime/pcaps",
            flows_dir="runtime/flows",
            batch_pcap_dir="runtime/_batch/pcaps",
            batch_flows_dir="runtime/_batch/flows",
        )

    def on_start(self):
        if self.worker is not None:
            return
        if self.detector is None:
            self._append_log("ERROR: detector is not loaded.")
            return

        cfg = self._make_config()
        self._append_log(f"[INFO] Starting with config: {asdict(cfg)}")
        self.status_var.set("Status: running")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        self.worker = IDSWorker(cfg=cfg, detector=self.detector, log_q=self.log_q, alert_q=self.alert_q)
        self.worker.start()

    def on_stop(self):
        if self.worker is None:
            return

        self._append_log("[INFO] Stopping IDS...")
        self.status_var.set("Status: stopping...")

        try:
            self.worker.stop()
        except Exception:
            pass

        self.worker = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Status: ready")

    def _poll_queues(self):
        try:
            while True:
                msg = self.log_q.get_nowait()
                self._append_log(msg)

                if msg.startswith("[OK] capture_"):
                    self.batch_count += 1
                    self.counters_var.set(f"Batches: {self.batch_count} | Alerts: {self.alert_count}")
        except queue.Empty:
            pass

        try:
            while True:
                a = self.alert_q.get_nowait()
                self.alert_count += 1
                self.tree.insert("", "end", values=(a.get("time"), f"{a.get('p_attack'):.4f}", a.get("source"), a.get("details")))
                self.counters_var.set(f"Batches: {self.batch_count} | Alerts: {self.alert_count}")
        except queue.Empty:
            pass

        self.after(200, self._poll_queues)


IDSApp = MainWindow
