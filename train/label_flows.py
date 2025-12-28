from __future__ import annotations

import os
import re
import glob
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Iterable, Optional

import numpy as np
import pandas as pd


DAY_RE = re.compile(r"(monday|tuesday|wednesday|thursday|friday)", re.IGNORECASE)

def _log(msg: str) -> None:
    print(msg, flush=True)

def _norm_cols(cols: Iterable[str]) -> List[str]:
    return [str(c).strip() for c in cols]

def _detect_day_from_name(name: str) -> Optional[str]:
    m = DAY_RE.search(name)
    return m.group(1).lower() if m else None

def _as_int64(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(np.int64).to_numpy()

def _as_str(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.strip().to_numpy(dtype=object)

def _ensure_cols(df: pd.DataFrame, needed: List[str], file_hint: str) -> None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{file_hint} missing columns: {missing}. Has: {list(df.columns)[:25]}...")

# ---------------------------
# CSV reading with encoding fallback
# ---------------------------

def _read_csv_head(path: str, nrows: int = 1, **kwargs) -> pd.DataFrame:
    """
    Read small head with encoding fallback (CICIDS2017 sometimes is not UTF-8).
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, nrows=nrows, low_memory=False, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise last_err  # type: ignore[misc]

def _read_csv_chunks(path: str, chunksize: int, **kwargs):
    """
    Read CSV in chunks with encoding fallback.
    IMPORTANT: UnicodeDecodeError may happen only during iteration,
    so we must actually try to pull the first chunk for each encoding.
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err: Optional[Exception] = None

    for enc in encodings:
        reader = None
        try:
            reader = pd.read_csv(
                path,
                chunksize=chunksize,
                low_memory=False,
                encoding=enc,
                **kwargs,
            )

            # Try to read first chunk to validate encoding
            first = next(reader)
            yield first

            # If first chunk succeeded, yield the rest
            for chunk in reader:
                yield chunk

            return  # success, stop trying encodings

        except UnicodeDecodeError as e:
            last_err = e
            # try next encoding
            continue
        except StopIteration:
            # empty file
            return
        except Exception as e:
            # If it's not a decode error, propagate (usually real CSV problem)
            raise

    raise last_err  # type: ignore[misc]


# ---------------------------
# Timestamp parsing (robust)
# ---------------------------

_TS_FORMATS = [
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %I:%M:%S %p",
    "%m/%d/%Y %I:%M %p",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
]

def _parse_ts_to_epoch_seconds(series: pd.Series) -> np.ndarray:
    """
    Converts Timestamp column to epoch seconds (int64).
    Tries known formats first, then falls back to pandas/dateutil.
    """
    s = series.astype(str).str.strip()

    ts = None
    for fmt in _TS_FORMATS:
        try:
            tmp = pd.to_datetime(s, format=fmt, errors="coerce")
            ts = tmp if ts is None else ts.fillna(tmp)
            if ts.notna().mean() > 0.98:
                break
        except Exception:
            continue

    if ts is None or ts.notna().sum() == 0:
        # final fallback; dayfirst=True helps with 02/07/2017 ambiguity
        ts = pd.to_datetime(s, errors="coerce", dayfirst=True)

    # Use astype instead of deprecated .view
    ns = ts.astype("int64", errors="ignore")
    if not np.issubdtype(ns.dtype, np.integer):
        out = np.zeros(len(s), dtype=np.int64)
        for i, v in enumerate(ts):
            out[i] = int(v.timestamp()) if pd.notna(v) else 0
        return out

    ns = ns.to_numpy(dtype=np.int64)
    ns = np.where(ns < 0, 0, ns)
    return (ns // 10**9).astype(np.int64)

# ---------------------------
# Protocol normalization
# ---------------------------

_PROTO_MAP = {"TCP": "6", "UDP": "17", "ICMP": "1"}

def _proto_norm(series: pd.Series) -> np.ndarray:
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().any():
        return num.fillna(0).astype(np.int64).astype(str).to_numpy(dtype=object)
    s = series.astype(str).str.strip().str.upper()
    return s.map(lambda x: _PROTO_MAP.get(x, x)).to_numpy(dtype=object)

# ---------------------------
# Schemas
# ---------------------------

REF = {
    "flow_id": "Flow ID",
    "src_ip": "Source IP",
    "dst_ip": "Destination IP",
    "src_port": "Source Port",
    "dst_port": "Destination Port",
    "proto": "Protocol",
    "ts": "Timestamp",
    "label": "Label",
}

GEN = {
    "flow_id": "Flow ID",
    "src_ip": "Src IP",
    "dst_ip": "Dst IP",
    "src_port": "Src Port",
    "dst_port": "Dst Port",
    "proto": "Protocol",
    "ts": "Timestamp",
    "label": "Label",
}

Key = Tuple[str, str, int, int, str]  # (src,dst,sp,dp,proto)

class RefIndex:
    """
    Two-level matching:
      1) Flow ID -> Label (primary)
      2) (src,dst,sp,dp,proto) + timestamp -> label (fallback; also reverse src/dst)
    """
    def __init__(self) -> None:
        self.flowid_to_label: Dict[str, str] = {}
        self._tmp: Dict[Key, List[Tuple[int, str]]] = {}
        self._final: Dict[Key, Tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _py_key(src: object, dst: object, sp: object, dp: object, pr: object) -> Key:
        return (str(src), str(dst), int(sp), int(dp), str(pr))

    def add_many_flowid(self, flow_ids: np.ndarray, labels: np.ndarray) -> None:
        # If duplicate Flow ID, prefer label != BENIGN
        for fid, lab in zip(flow_ids, labels):
            f = str(fid).strip()
            l = str(lab).strip()
            if not f or f.lower() == "nan":
                continue
            prev = self.flowid_to_label.get(f)
            if prev is None:
                self.flowid_to_label[f] = l
            else:
                if prev.upper() == "BENIGN" and l.upper() != "BENIGN":
                    self.flowid_to_label[f] = l

    def add_many_tuplets(
        self,
        src: np.ndarray, dst: np.ndarray,
        sp: np.ndarray, dp: np.ndarray,
        pr: np.ndarray,
        ts_s: np.ndarray,
        labels: np.ndarray
    ) -> None:
        for s, d, spv, dpv, prv, t, lab in zip(src, dst, sp, dp, pr, ts_s, labels):
            tt = int(t)
            if tt <= 0:
                continue
            k = self._py_key(s, d, spv, dpv, prv)
            self._tmp.setdefault(k, []).append((tt, str(lab)))

    def finalize(self) -> None:
        out: Dict[Key, Tuple[np.ndarray, np.ndarray]] = {}
        for k, lst in self._tmp.items():
            lst.sort(key=lambda x: x[0])
            ts = np.array([x[0] for x in lst], dtype=np.int64)
            labs = np.array([x[1] for x in lst], dtype=object)
            out[k] = (ts, labs)
        self._final = out
        self._tmp.clear()

    def _match_one(self, key: Key, ts_s: int, tolerance_s: int) -> Optional[str]:
        pair = self._final.get(key)
        if pair is None:
            return None
        ts_arr, lab_arr = pair
        if ts_arr.size == 0:
            return None

        i = int(np.searchsorted(ts_arr, ts_s))
        best_lab = None
        best_dt = None

        for j in (i - 1, i, i + 1):
            if 0 <= j < ts_arr.size:
                dt = abs(int(ts_arr[j]) - int(ts_s))
                if dt <= tolerance_s and (best_dt is None or dt < best_dt):
                    best_dt = dt
                    best_lab = str(lab_arr[j])
        return best_lab

    def match_bidir(self, key: Key, ts_s: int, tolerance_s: int) -> Optional[str]:
        lab = self._match_one(key, ts_s, tolerance_s)
        if lab:
            return lab
        src, dst, sp, dp, pr = key
        rkey: Key = (dst, src, dp, sp, pr)
        return self._match_one(rkey, ts_s, tolerance_s)

def build_ref_index_for_day(
    ref_files: List[str],
    day: str,
    chunksize: int,
    max_rows: int,
) -> RefIndex:
    idx = RefIndex()

    usecols = [
        REF["flow_id"], REF["src_ip"], REF["dst_ip"],
        REF["src_port"], REF["dst_port"], REF["proto"], REF["ts"], REF["label"],
    ]

    total = 0
    for fp in ref_files:
        base = os.path.basename(fp)
        _log(f"[+] Ref[{day}] reading: {base}")

        for chunk in _read_csv_chunks(fp, chunksize=chunksize):
            chunk.columns = _norm_cols(chunk.columns)
            _ensure_cols(chunk, usecols, base)

            flow_ids = _as_str(chunk[REF["flow_id"]])
            lab = _as_str(chunk[REF["label"]])
            idx.add_many_flowid(flow_ids, lab)

            src = _as_str(chunk[REF["src_ip"]])
            dst = _as_str(chunk[REF["dst_ip"]])
            sp  = _as_int64(chunk[REF["src_port"]])
            dp  = _as_int64(chunk[REF["dst_port"]])
            pr  = _proto_norm(chunk[REF["proto"]])
            ts_s = _parse_ts_to_epoch_seconds(chunk[REF["ts"]])

            idx.add_many_tuplets(src, dst, sp, dp, pr, ts_s, lab)

            total += len(chunk)
            if max_rows > 0 and total >= max_rows:
                _log(f"[i] Ref[{day}] reached max_rows={max_rows}, stopping early.")
                break

        if max_rows > 0 and total >= max_rows:
            break

    _log(f"[*] Ref[{day}] building index...")
    idx.finalize()
    _log(f"[✓] Ref[{day}] indexed flowid={len(idx.flowid_to_label)} keys={len(idx._final)} rows≈{total}")
    return idx

def label_generated_file(
    gen_csv: str,
    out_csv: str,
    ref_index: RefIndex,
    tolerance_s: int,
    chunksize: int,
    unknown_label: str = "BENIGN",
) -> Tuple[int, int, int, int, int]:
    """
    Returns:
      total_rows, non_benign, matched_flowid, matched_fallback, ts_zero
    """
    base = os.path.basename(gen_csv)

    head = _read_csv_head(gen_csv, nrows=1)
    head.columns = _norm_cols(head.columns)

    needed = [
        GEN["flow_id"], GEN["src_ip"], GEN["dst_ip"],
        GEN["src_port"], GEN["dst_port"], GEN["proto"], GEN["ts"],
    ]
    _ensure_cols(head, needed, base)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrote_header = False
    total = 0
    non_benign = 0
    matched_flowid = 0
    matched_fallback = 0
    ts_zero = 0

    for chunk in _read_csv_chunks(gen_csv, chunksize=chunksize):
        chunk.columns = _norm_cols(chunk.columns)
        _ensure_cols(chunk, needed, base)

        flow_ids = _as_str(chunk[GEN["flow_id"]])

        src = _as_str(chunk[GEN["src_ip"]])
        dst = _as_str(chunk[GEN["dst_ip"]])
        sp  = _as_int64(chunk[GEN["src_port"]])
        dp  = _as_int64(chunk[GEN["dst_port"]])
        pr  = _proto_norm(chunk[GEN["proto"]])
        ts_s = _parse_ts_to_epoch_seconds(chunk[GEN["ts"]])

        ts_zero += int(np.sum(ts_s <= 0))

        out_labels: List[str] = []

        for fid, s, d, spv, dpv, prv, t in zip(flow_ids, src, dst, sp, dp, pr, ts_s):
            f = str(fid).strip()
            lab = ref_index.flowid_to_label.get(f)

            if lab is not None:
                matched_flowid += 1
            else:
                key: Key = (str(s), str(d), int(spv), int(dpv), str(prv))
                lab = ref_index.match_bidir(key, int(t), tolerance_s=tolerance_s)
                if lab is not None:
                    matched_fallback += 1
                else:
                    lab = unknown_label

            out_labels.append(str(lab))

        chunk[GEN["label"]] = out_labels

        total += len(chunk)
        non_benign += int(np.sum(np.array(out_labels, dtype=object) != "BENIGN"))

        chunk.to_csv(out_csv, mode="a", index=False, header=(not wrote_header))
        wrote_header = True

    return total, non_benign, matched_flowid, matched_fallback, ts_zero

def main(
    pcap_flows_dir: str,
    ref_dir: str,
    out_dir: str,
    tolerance_s: int,
    chunksize: int,
    max_ref_rows: int,
    unknown_label: str,
) -> None:
    pcap_flows_dir = os.path.abspath(pcap_flows_dir)
    ref_dir = os.path.abspath(ref_dir)
    out_dir = os.path.abspath(out_dir)

    gen_files = sorted(glob.glob(os.path.join(pcap_flows_dir, "*.csv")))
    if not gen_files:
        raise RuntimeError(f"No generated flow CSV files in: {pcap_flows_dir}")

    ref_files = sorted(glob.glob(os.path.join(ref_dir, "*.csv")))
    if not ref_files:
        raise RuntimeError(f"No reference TrafficLabelling CSV files in: {ref_dir}")

    ref_by_day: Dict[str, List[str]] = {d: [] for d in ["monday", "tuesday", "wednesday", "thursday", "friday"]}
    for fp in ref_files:
        day = _detect_day_from_name(os.path.basename(fp))
        if day in ref_by_day:
            ref_by_day[day].append(fp)

    for d, files in ref_by_day.items():
        if files:
            _log(f"[+] Day={d}: reference files={len(files)}")

    ref_index_cache: Dict[str, RefIndex] = {}
    os.makedirs(out_dir, exist_ok=True)

    for gen_fp in gen_files:
        base = os.path.basename(gen_fp)
        day = _detect_day_from_name(base)
        if day is None:
            _log(f"[!] Skipping {base}: cannot detect day from filename")
            continue

        day_refs = ref_by_day.get(day, [])
        if not day_refs:
            _log(f"[!] Skipping {base}: no reference files for day={day} in {ref_dir}")
            continue

        if day not in ref_index_cache:
            _log(f"[i] Building reference index for day={day} (ONLY this day will be used)")
            ref_index_cache[day] = build_ref_index_for_day(
                ref_files=day_refs,
                day=day,
                chunksize=chunksize,
                max_rows=max_ref_rows,
            )

        out_fp = os.path.join(out_dir, base.replace(".csv", ".labeled.csv"))
        _log(f"[*] Labeling {base} -> {os.path.basename(out_fp)} (tolerance_s={tolerance_s}, unknown_label={unknown_label})")

        total, non_benign, m_flowid, m_fb, tz = label_generated_file(
            gen_csv=gen_fp,
            out_csv=out_fp,
            ref_index=ref_index_cache[day],
            tolerance_s=tolerance_s,
            chunksize=chunksize,
            unknown_label=unknown_label,
        )
        _log(
            f"[✓] {base}: rows={total} "
            f"matched_flowid≈{m_flowid} matched_fallback≈{m_fb} ts_zero≈{tz} non_benign≈{non_benign} -> {out_fp}"
        )

    _log(f"[✓] Done. Labeled files in: {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap_flows_dir", required=True)
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tolerance_s", type=int, default=600)
    ap.add_argument("--chunksize", type=int, default=200000)
    ap.add_argument("--max_ref_rows", type=int, default=0)
    ap.add_argument("--unknown_label", default="BENIGN")
    args = ap.parse_args()

    main(
        pcap_flows_dir=args.pcap_flows_dir,
        ref_dir=args.ref_dir,
        out_dir=args.out_dir,
        tolerance_s=args.tolerance_s,
        chunksize=args.chunksize,
        max_ref_rows=args.max_ref_rows,
        unknown_label=args.unknown_label,
    )
