from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List

MAIN_CLASS = "cic.cs.unb.ca.ifm.CICFlowMeter"


def _ensure_trailing_sep(p: str) -> str:
    p = str(p).strip().strip('"')
    if not p.endswith(("\\", "/")):
        p = p + os.sep
    return p


def _build_classpath(cicflowmeter_jar: str, cicflow_lib_dir: str | None, jnetpcap_jar: str) -> str:
    """
    Classpath = główny jar CICFlowMeter + wszystkie jar-y z lib/ + jnetpcap.jar
    """
    jars: List[str] = []

    jars.append(str(Path(cicflowmeter_jar).resolve()))

    if cicflow_lib_dir:
        lib = Path(cicflow_lib_dir).resolve()
        if lib.exists() and lib.is_dir():
            for j in sorted(lib.glob("*.jar")):
                jars.append(str(j))

    jars.append(str(Path(jnetpcap_jar).resolve()))

    # Windows: ';'  Linux/Mac: ':'
    return os.pathsep.join(jars)


def run_cicflowmeter(
    java_bin: str,
    cicflowmeter_jar: str,
    jnetpcap_jar: str,
    jnetpcap_dll_dir: str,
    pcap_dir: str,
    out_dir: str,
    cicflow_lib_dir: str | None = None,  # <-- NOWE (np. tools/cicflowmeter/lib)
) -> None:
    """
    Uruchamia CICFlowMeter na folderze z .pcap -> CSV w out_dir.
    Wymusza końcowy separator ścieżek (ważne na Windows dla niektórych buildów CICFlowMeter).

    UWAGA: Wiele buildów CICFlowMeter wymaga dodatkowych zależności (np. slf4j) w classpath,
    dlatego dorzucamy wszystkie *.jar z cicflow_lib_dir.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pcap_dir_abs = _ensure_trailing_sep(str(Path(pcap_dir).resolve()))
    out_dir_abs = _ensure_trailing_sep(str(out_path.resolve()))

    cp = _build_classpath(
        cicflowmeter_jar=cicflowmeter_jar,
        cicflow_lib_dir=cicflow_lib_dir,
        jnetpcap_jar=jnetpcap_jar,
    )

    cmd = [
        java_bin,
        f"-Djava.library.path={str(Path(jnetpcap_dll_dir).resolve())}",
        "-cp",
        cp,
        MAIN_CLASS,
        pcap_dir_abs,
        out_dir_abs,
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        raise RuntimeError(
            "CICFlowMeter failed.\n"
            f"Return code: {e.returncode}\n"
            f"Command: {cmd}\n\n"
            "---- STDOUT ----\n"
            f"{stdout}\n\n"
            "---- STDERR ----\n"
            f"{stderr}"
        ) from e


def newest_csv(out_dir: str) -> Path:
    out = Path(out_dir)
    csvs = sorted(out.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        raise RuntimeError("CICFlowMeter nie wygenerował żadnego CSV w folderze flows.")
    return csvs[0]
