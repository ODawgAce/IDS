from __future__ import annotations
import subprocess
from pathlib import Path

def start_dumpcap(dumpcap_path: str, interface: str, out_dir: str, rotate_seconds: int) -> subprocess.Popen:
    """
    Ring-buffer PCAP: tworzy kolejne pliki co rotate_seconds w formacie pcap.
    (Twoja wersja CICFlowMeter czyta *.pcap, nie pcapng)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -b duration:5 -> nowy plik co 5 sekund
    # -F pcap      -> wymusza format pcap (dumpcap -P jest deprecated)
    cmd = [
        dumpcap_path,
        "-i", str(interface),
        "-b", f"duration:{rotate_seconds}",
        "-F", "pcap",
        "-w", str(out / "capture.pcap"),
        "-q",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)