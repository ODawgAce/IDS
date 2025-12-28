from __future__ import annotations

import os
import argparse
import subprocess
from pathlib import Path
from typing import List


def run_cicflow(cicflow_jar: str, jnetpcap_dir: str, in_dir: str, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "java",
        f"-Djava.library.path={jnetpcap_dir}",
        "-cp",
        f"{cicflow_jar};{os.path.join(jnetpcap_dir, 'jnetpcap.jar')}",
        "cic.cs.unb.ca.ifm.CICFlowMeter",
        str(Path(in_dir)) + os.sep,
        str(Path(out_dir)) + os.sep,
    ]
    print(f"[*] CICFlowMeter in={in_dir} out={out_dir}", flush=True)
    subprocess.run(cmd, check=True)


def _try_link(dst: Path, src: Path) -> bool:
    """
    Try hardlink first (no extra disk), then symlink.
    Returns True if created.
    """
    try:
        if dst.exists():
            dst.unlink()
    except Exception:
        pass

    # hardlink (best)
    try:
        os.link(str(src), str(dst))
        return True
    except Exception:
        pass

    # symlink (may require admin/dev-mode)
    try:
        os.symlink(str(src), str(dst))
        return True
    except Exception:
        return False


def _chunk(items: List[Path], batch_size: int) -> List[List[Path]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def main(pcap_dir: str, out_dir: str, cicflow_jar: str, jnetpcap_dir: str, batch_size: int) -> None:
    pcap_dir_p = Path(pcap_dir)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    pcaps = sorted([p for p in pcap_dir_p.iterdir() if p.is_file() and p.suffix.lower() in [".pcap", ".pcapng"]])
    if not pcaps:
        raise SystemExit(f"No PCAP/PCAPNG found in: {pcap_dir}")

    # Jeśli batch_size == 0 -> uruchom CICFlowMeter bezpośrednio na katalogu PCAP (zero kopiowania)
    if batch_size == 0:
        print("[+] Running directly on pcap_dir (no batch folder, no copies).", flush=True)
        run_cicflow(cicflow_jar=cicflow_jar, jnetpcap_dir=jnetpcap_dir, in_dir=str(pcap_dir_p), out_dir=str(out_dir_p))
        print(f"[✓] Flows generated into: {out_dir_p}", flush=True)
        return

    # batch folder (tylko linki!)
    batch_in = out_dir_p / "_batch_in"
    batch_in.mkdir(parents=True, exist_ok=True)

    batches = _chunk(pcaps, batch_size)

    print(f"[+] Found {len(pcaps)} pcaps. Processing in {len(batches)} batch(es), batch_size={batch_size}", flush=True)

    for bi, batch in enumerate(batches, start=1):
        # wyczyść batch_in
        for f in batch_in.glob("*"):
            try:
                f.unlink()
            except Exception:
                pass

        print(f"[*] Batch {bi}/{len(batches)}: preparing {len(batch)} files (links only)...", flush=True)

        linked_ok = 0
        for p in batch:
            dst = batch_in / p.name
            if _try_link(dst, p):
                linked_ok += 1
            else:
                # jeśli nie da się linkować, nie kopiujemy (bo brak miejsca) -> fallback: uruchom bezpośrednio na pcap_dir
                print(f"[WARN] Cannot link {p.name}. Falling back to direct mode (batch_size=0).", flush=True)
                run_cicflow(cicflow_jar=cicflow_jar, jnetpcap_dir=jnetpcap_dir, in_dir=str(pcap_dir_p), out_dir=str(out_dir_p))
                print(f"[✓] Flows generated into: {out_dir_p}", flush=True)
                return

        print(f"[+] Linked {linked_ok}/{len(batch)} files. Running CICFlowMeter...", flush=True)
        run_cicflow(cicflow_jar=cicflow_jar, jnetpcap_dir=jnetpcap_dir, in_dir=str(batch_in), out_dir=str(out_dir_p))

    print(f"[✓] Flows generated into: {out_dir_p}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cicflow_jar", required=True)
    ap.add_argument("--jnetpcap_dir", required=True)
    ap.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="How many pcaps per CICFlowMeter run. Use 0 to run directly on pcap_dir (no batch folder).",
    )
    args = ap.parse_args()

    main(args.pcap_dir, args.out_dir, args.cicflow_jar, args.jnetpcap_dir, args.batch_size)
