from __future__ import annotations

import os
import queue
import shutil
import socket
import subprocess
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from tkinter import ttk, messagebox

import psutil

from app.detector import HybridDetector
from app.worker import IDSWorker, WorkerConfig
from app.path_utils import p, ensure_data_dirs
from app.logging_utils import setup_logger


# =========================================================
# Helpers
# =========================================================

def _is_ipv4(addr: str) -> bool:
    try:
        socket.inet_aton(addr)
        return True
    except OSError:
        return False


def list_interfaces_pretty() -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    addrs = psutil.net_if_addrs()
    stats = psutil.net_if_stats()

    for name, addr_list in addrs.items():
        st = stats.get(name)
        is_up = bool(st.isup) if st else False

        ipv4 = ""
        for a in addr_list:
            if getattr(a, "family", None) == socket.AF_INET and a.address and _is_ipv4(a.address):
                if a.address != "127.0.0.1":
                    ipv4 = a.address
                    break
                ipv4 = a.address

        tag = "UP" if is_up else "DOWN"
        display = f"{name} — {ipv4} ({tag})" if ipv4 else f"{name} — (no IPv4) ({tag})"
        out.append((display, name, ipv4))

    out.sort(key=lambda t: (0 if "(UP)" in t[0] else 1, t[0].lower()))
    return out


def find_dumpcap() -> str | None:
    env = os.environ.get("WIRESHARK_DUMPCAP")
    if env and Path(env).exists():
        return env

    w = shutil.which("dumpcap")
    if w and Path(w).exists():
        return w

    for c in (
        r"C:\Program Files\Wireshark\dumpcap.exe",
        r"C:\Program Files (x86)\Wireshark\dumpcap.exe",
    ):
        if Path(c).exists():
            return c
    return None


# =========================================================
# GUI
# =========================================================

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("IDS (RF + LSTM)")
        self.geometry("1100x650")

        # ---- AppData dirs ----
        self.dirs = ensure_data_dirs()
        self.artifacts_root: Path = self.dirs["artifacts"]
        self.runtime_root: Path = self.dirs["runtime"]

        self.logger = setup_logger(str(self.dirs["logs"] / "app.log"))
        self.logger.info("Start GUI")

        self.detector: HybridDetector | None = None
        self.worker: IDSWorker | None = None

        self.log_q: queue.Queue[str] = queue.Queue()
        self.alert_q: queue.Queue[dict] = queue.Queue()

        self.batch_count = 0
        self.alert_count = 0
        self.models_ready = False

        self._iface_map: dict[str, str] = {}

        self._build_ui()
        self._populate_interfaces()
        self._try_load_models()
        self._auto_clean_batch_on_start()

        self.after(200, self._poll_queues)

    # =====================================================
    # UI
    # =====================================================

    def _build_ui(self):
        frm = ttk.LabelFrame(self, text="Sterowanie")
        frm.pack(fill="x", padx=10, pady=10)

        row = ttk.Frame(frm)
        row.pack(fill="x", padx=10, pady=8)

        ttk.Label(row, text="Interfejs:").pack(side="left")

        self.iface_var = tk.StringVar(value="")
        self.iface_box = ttk.Combobox(
            row, textvariable=self.iface_var, width=55, state="readonly")
        self.iface_box.pack(side="left", padx=8)

        self.refresh_btn = ttk.Button(
            row, text="Odśwież", command=self._populate_interfaces)
        self.refresh_btn.pack(side="left")

        row2 = ttk.Frame(frm)
        row2.pack(fill="x", padx=10, pady=8)

        ttk.Label(row2, text="Próg alertu:").pack(side="left")
        self.thr_var = tk.DoubleVar(value=0.50)
        ttk.Spinbox(row2, from_=0, to=1, increment=0.01,
                    textvariable=self.thr_var, width=8).pack(side="left", padx=8)

        ttk.Label(row2, text="Czas batcha (s):").pack(
            side="left", padx=(20, 0))
        self.cap_var = tk.IntVar(value=60)
        ttk.Spinbox(row2, from_=5, to=180, increment=5,
                    textvariable=self.cap_var, width=6).pack(side="left", padx=8)

        row3 = ttk.Frame(frm)
        row3.pack(fill="x", padx=10, pady=(0, 10))

        self.start_btn = ttk.Button(row3, text="Start", command=self.on_start)
        self.start_btn.pack(side="left", expand=True, fill="x", padx=5)

        self.stop_btn = ttk.Button(
            row3, text="Stop", command=self.on_stop, state="disabled")
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=5)

        self.clear_btn = ttk.Button(
            row3, text="Wyczyść runtime", command=self.on_clear_runtime)
        self.clear_btn.pack(side="left", expand=True, fill="x", padx=5)

        self.download_btn = ttk.Button(
            row3, text="Pobierz modele", command=self.on_download_models)
        self.download_btn.pack(side="left", expand=True, fill="x", padx=5)

        stat = ttk.Frame(self)
        stat.pack(fill="x", padx=10)

        self.status_var = tk.StringVar(value="Status: idle")
        ttk.Label(stat, textvariable=self.status_var).pack(side="left")

        self.counters_var = tk.StringVar(value="Batches: 0 | Alerts: 0")
        ttk.Label(stat, textvariable=self.counters_var).pack(side="right")

        log_frm = ttk.LabelFrame(self, text="Log")
        log_frm.pack(fill="both", expand=True, padx=10, pady=10)

        self.log_text = tk.Text(log_frm, height=10)
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

        alert_frm = ttk.LabelFrame(self, text="Alerty")
        alert_frm.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        cols = ("time", "p_attack", "source", "details")
        self.tree = ttk.Treeview(alert_frm, columns=cols, show="headings")
        for c in cols:
            self.tree.heading(c, text=c)
        self.tree.pack(fill="both", expand=True)

    # =====================================================
    # MODELS
    # =====================================================

    def _try_load_models(self):
        try:
            self.detector = HybridDetector(
                rf_path=str(self.artifacts_root / "rf.joblib"),
                lstm_path=str(self.artifacts_root / "lstm.keras.best.keras"),
                preproc_path=str(self.artifacts_root / "preproc.joblib"),
                seq_len=20,
            )
            self.models_ready = True
            self.status_var.set("Status: ready")
            self._append_log("[OK] Models loaded.")

        except Exception as e:
            self.models_ready = False
            self.detector = None
            self.status_var.set("Status: models missing")
            self._append_log(f"[ERROR] Models not loaded: {e}")

    def on_download_models(self):
        ps1 = Path(p("scripts", "download_models.ps1"))
        subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File",
                str(ps1), "-TargetDir", str(self.artifacts_root)],
            check=False,
        )
        self._try_load_models()

    # =====================================================
    # REST – runtime, worker, queues
    # =====================================================

    def _append_log(self, line: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self.logger.info(line)

    def _populate_interfaces(self):
        items = list_interfaces_pretty()
        self._iface_map = {d: n for d, n, _ in items}
        self.iface_box["values"] = list(self._iface_map.keys())
        if items:
            self.iface_var.set(items[0][0])

    def _auto_clean_batch_on_start(self):
        for d in (
            self.runtime_root / "_batch" / "pcaps",
            self.runtime_root / "_batch" / "flows",
        ):
            d.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)

    def on_clear_runtime(self):
        if self.worker:
            return
        shutil.rmtree(self.runtime_root, ignore_errors=True)
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self._append_log("[OK] Runtime cleaned")

    def on_start(self):
        if not self.models_ready:
            messagebox.showerror("Błąd", "Modele niezaładowane")
            return

        iface = self._iface_map.get(self.iface_var.get())
        dumpcap = find_dumpcap()
        if not iface or not dumpcap:
            messagebox.showerror("Błąd", "Brak dumpcap lub interfejsu")
            return

        cfg = WorkerConfig(
            dumpcap_path=dumpcap,
            java_bin="java",
            cicflow_jar=p("tools", "cicflowmeter",
                          "CICFlowMeterV3-0.0.4-SNAPSHOT.jar"),
            jnetpcap_jar=p("tools", "jnetpcap", "jnetpcap.jar"),
            jnetpcap_dll_dir=p("tools", "jnetpcap"),
            interface=iface,
            capture_seconds=int(self.cap_var.get()),
            threshold=float(self.thr_var.get()),
            seq_len=20,
            runtime_dir=str(self.runtime_root),
            pcap_dir=str(self.runtime_root / "pcaps"),
            flows_dir=str(self.runtime_root / "flows"),
            batch_pcap_dir=str(self.runtime_root / "_batch" / "pcaps"),
            batch_flows_dir=str(self.runtime_root / "_batch" / "flows"),
        )

        self.worker = IDSWorker(cfg, self.detector, self.log_q, self.alert_q)
        self.worker.start()

        # --- NOWY KOD: Zarządzanie stanem przycisków ---
        self.start_btn.configure(state="disabled")   # Zablokuj Start
        self.stop_btn.configure(state="normal")      # Odblokuj Stop
        # Zablokuj czyszczenie podczas pracy
        self.clear_btn.configure(state="disabled")
        self.status_var.set("Status: Running...")
        self._append_log("[INFO] Worker started.")

    def on_stop(self):
        if self.worker:
            self.worker.stop()
            self.worker = None

        # --- NOWY KOD: Przywracanie przycisków ---
        self.stop_btn.configure(state="disabled")    # Zablokuj Stop
        self.start_btn.configure(state="normal")     # Odblokuj Start
        self.clear_btn.configure(state="normal")     # Odblokuj czyszczenie
        self.status_var.set("Status: Stopped")
        self._append_log("[INFO] Worker stopped.")

    def _poll_queues(self):
        # 1. Odbieranie LOGÓW (To już było)
        try:
            while True:
                msg = self.log_q.get_nowait()
                self._append_log(msg)

                # Mały bonus: Zliczanie batchy na podstawie logów
                if "Generating flows" in msg:
                    self.batch_count += 1
                    self.counters_var.set(
                        f"Batches: {self.batch_count} | Alerts: {self.alert_count}")
        except queue.Empty:
            pass

        # 2. Odbieranie ALERTÓW (Tego brakowało!)
        try:
            while True:
                # Wyciągnij alert z kolejki
                alert = self.alert_q.get_nowait()
                self.alert_count += 1

                # Przygotuj dane do tabeli
                # alert["p_attack"] może być numpy.float, więc rzutujemy na float
                p_val = float(alert["p_attack"])

                values = (
                    alert["time"],
                    f"{p_val:.4f}",     # Formatowanie np. 0.9982
                    alert["source"],
                    alert["details"]
                )

                # Wstaw na samą górę tabeli (index 0)
                self.tree.insert("", 0, values=values)

                # Zaktualizuj licznik na dole
                self.counters_var.set(
                    f"Batches: {self.batch_count} | Alerts: {self.alert_count}")

        except queue.Empty:
            pass

        # Zapętl funkcję co 200ms
        self.after(200, self._poll_queues)


IDSApp = MainWindow
