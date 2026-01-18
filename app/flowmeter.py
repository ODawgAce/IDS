from __future__ import annotations

import os
import subprocess
from pathlib import Path

MAIN_CLASS = "cic.cs.unb.ca.ifm.CICFlowMeter"


def _ensure_trailing_sep(p: str) -> str:
    """
    CICFlowMeter (szczególnie na Windows) potrafi sklejać folder+plik bez separatora,
    jeśli folder nie kończy się na \\ lub /.
    """
    if not p:
        return p
    if p.endswith("\\") or p.endswith("/"):
        return p
    return p + os.sep


def run_cicflowmeter(
    java_bin: str,
    cicflowmeter_jar: str,
    jnetpcap_jar: str,
    jnetpcap_dll_dir: str,
    pcap_dir: str,
    out_dir: str,
) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    pcap_dir = _ensure_trailing_sep(str(pcap_dir))
    out_dir = _ensure_trailing_sep(str(out_dir))

    cmd = [
        java_bin,
        f"-Djava.library.path={jnetpcap_dll_dir}",
        "-cp",
        f"{cicflowmeter_jar};{jnetpcap_jar}",
        MAIN_CLASS,
        pcap_dir,
        out_dir,
    ]
    subprocess.run(cmd, check=True)


def newest_csv(out_dir: str) -> Path:
    out = Path(out_dir)
    csvs = sorted(out.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        raise RuntimeError("CICFlowMeter nie wygenerował żadnego CSV w folderze flows.")
    return csvs[0]