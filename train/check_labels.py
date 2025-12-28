from __future__ import annotations

import os, re, glob, argparse
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

DAY_RE = re.compile(r"(monday|tuesday|wednesday|thursday|friday)", re.IGNORECASE)

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

_PROTO_MAP = {"TCP": "6", "UDP": "17", "ICMP": "1"}

Key = Tuple[str, str, int, int, str]

def _detect_day(name: str) -> Optional[str]:
    m = DAY_RE.search(name)
    return m.group(1).lower() if m else None

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _as_int64(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(np.int64).to_numpy()

def _as_str(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.strip().to_numpy(dtype=object)

def _proto_norm(series: pd.Series) -> np.ndarray:
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().any():
        return num.fillna(0).astype(np.int64).astype(str).to_numpy(dtype=object)
    s = series.astype(str).str.strip().str.upper()
    return s.map(lambda x: _PROTO_MAP.get(x, x)).to_numpy(dtype=object)

def _parse_ts_to_epoch_seconds(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.strip()
    # u Ciebie działało dayfirst=True na wygenerowanych
    ts = pd.to_datetime(s, errors="coerce", dayfirst=True)
    ns = ts.astype("int64", errors="ignore")
    if not np.issubdtype(ns.dtype, np.integer):
        out = np.zeros(len(s), dtype=np.int64)
        for i, v in enumerate(ts):
            out[i] = int(v.timestamp()) if pd.notna(v) else 0
        return out
    ns = ns.to_numpy(dtype=np.int64)
    ns = np.where(ns < 0, 0, ns)
    return (ns // 10**9).astype(np.int64)

@dataclass
class RefIndex:
    flowid_to_label: Dict[str, str]
    tuple_to_ts_lab: Dict[Key, Tuple[np.ndarray, np.ndarray]]

def build_ref_index(ref_files: List[str], chunksize: int = 200000) -> RefIndex:
    flowid_to_label: Dict[str, str] = {}
    tmp: Dict[Key, List[Tuple[int, str]]] = {}

    usecols = [REF["flow_id"], REF["src_ip"], REF["dst_ip"], REF["src_port"], REF["dst_port"], REF["proto"], REF["ts"], REF["label"]]

    for fp in ref_files:
        for chunk in pd.read_csv(fp, chunksize=chunksize, low_memory=False, encoding_errors="replace"):
            chunk = _norm_cols(chunk)
            missing = [c for c in usecols if c not in chunk.columns]
            if missing:
                raise ValueError(f"{os.path.basename(fp)} missing columns: {missing}")

            fid = _as_str(chunk[REF["flow_id"]])
            lab = _as_str(chunk[REF["label"]])

            # flowid->label (prefer non-benign)
            for f, l in zip(fid, lab):
                f = str(f).strip()
                l = str(l).strip()
                if not f or f.lower() == "nan":
                    continue
                prev = flowid_to_label.get(f)
                if prev is None:
                    flowid_to_label[f] = l
                else:
                    if prev.upper() == "BENIGN" and l.upper() != "BENIGN":
                        flowid_to_label[f] = l

            src = _as_str(chunk[REF["src_ip"]])
            dst = _as_str(chunk[REF["dst_ip"]])
            sp  = _as_int64(chunk[REF["src_port"]])
            dp  = _as_int64(chunk[REF["dst_port"]])
            pr  = _proto_norm(chunk[REF["proto"]])
            ts  = _parse_ts_to_epoch_seconds(chunk[REF["ts"]])

            for s, d, spv, dpv, prv, t, l in zip(src, dst, sp, dp, pr, ts, lab):
                tt = int(t)
                if tt <= 0:
                    continue
                k: Key = (str(s), str(d), int(spv), int(dpv), str(prv))
                tmp.setdefault(k, []).append((tt, str(l)))

    # finalize
    out: Dict[Key, Tuple[np.ndarray, np.ndarray]] = {}
    for k, lst in tmp.items():
        lst.sort(key=lambda x: x[0])
        ts = np.array([x[0] for x in lst], dtype=np.int64)
        labs = np.array([x[1] for x in lst], dtype=object)
        out[k] = (ts, labs)

    return RefIndex(flowid_to_label=flowid_to_label, tuple_to_ts_lab=out)

def _match_one(ref: RefIndex, key: Key, ts_s: int, tol: int) -> Optional[str]:
    pair = ref.tuple_to_ts_lab.get(key)
    if pair is None:
        return None
    ts_arr, lab_arr = pair
    if ts_arr.size == 0:
        return None
    i = int(np.searchsorted(ts_arr, ts_s))
    best = None
    best_dt = None
    for j in (i-1, i, i+1):
        if 0 <= j < ts_arr.size:
            dt = abs(int(ts_arr[j]) - int(ts_s))
            if dt <= tol and (best_dt is None or dt < best_dt):
                best_dt = dt
                best = str(lab_arr[j])
    return best

def match_bidir(ref: RefIndex, key: Key, ts_s: int, tol: int) -> Optional[str]:
    lab = _match_one(ref, key, ts_s, tol)
    if lab:
        return lab
    src, dst, sp, dp, pr = key
    rkey: Key = (dst, src, dp, sp, pr)
    return _match_one(ref, rkey, ts_s, tol)

def check_one(labeled_csv: str, ref_dir: str, sample: int, seed: int, tolerance_s: int) -> Tuple[str, int, int, float, int, int, int]:
    base = os.path.basename(labeled_csv)
    day = _detect_day(base)
    if not day:
        raise RuntimeError(f"Cannot detect day from {base}")

    ref_files = sorted(glob.glob(os.path.join(ref_dir, f"{day.capitalize()}*.csv"))) + sorted(glob.glob(os.path.join(ref_dir, f"{day}*.csv")))
    # pewniejsze: bierz wszystkie i filtruj regexem po dniu
    if not ref_files:
        ref_files = []
        for f in glob.glob(os.path.join(ref_dir, "*.csv")):
            if _detect_day(os.path.basename(f)) == day:
                ref_files.append(f)
        ref_files.sort()

    ref = build_ref_index(ref_files)

    # sample from labeled
    df = pd.read_csv(labeled_csv, low_memory=False)
    df = _norm_cols(df)
    need = [GEN["flow_id"], GEN["src_ip"], GEN["dst_ip"], GEN["src_port"], GEN["dst_port"], GEN["proto"], GEN["ts"], GEN["label"]]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{base} missing columns: {missing}")

    if len(df) > sample:
        df = df.sample(n=sample, random_state=seed)

    flow_ids = _as_str(df[GEN["flow_id"]])
    src = _as_str(df[GEN["src_ip"]])
    dst = _as_str(df[GEN["dst_ip"]])
    sp  = _as_int64(df[GEN["src_port"]])
    dp  = _as_int64(df[GEN["dst_port"]])
    pr  = _proto_norm(df[GEN["proto"]])
    ts  = _parse_ts_to_epoch_seconds(df[GEN["ts"]])

    y = _as_str(df[GEN["label"]])

    with_ref = 0
    mism = 0
    used_flowid = 0
    used_fallback = 0

    for fid, s, d, spv, dpv, prv, t, lab in zip(flow_ids, src, dst, sp, dp, pr, ts, y):
        fid = str(fid).strip()
        lab = str(lab).strip()
        ref_lab = ref.flowid_to_label.get(fid)
        if ref_lab is not None:
            used_flowid += 1
            with_ref += 1
            if str(ref_lab).strip() != lab:
                mism += 1
        else:
            key: Key = (str(s), str(d), int(spv), int(dpv), str(prv))
            ref_lab = match_bidir(ref, key, int(t), tolerance_s)
            if ref_lab is not None:
                used_fallback += 1
                with_ref += 1
                if str(ref_lab).strip() != lab:
                    mism += 1

    agree = 1.0 if with_ref == 0 else (with_ref - mism) / with_ref
    return base, len(df), with_ref, float(agree), mism, used_flowid, used_fallback

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled_dir", required=True)
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tolerance_s", type=int, default=600)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.labeled_dir, "*.csv")))
    print(f"[i] Files: {len(files)}\n", flush=True)

    total_rows = 0
    total_with_ref = 0
    total_mism = 0
    total_flowid = 0
    total_fallback = 0

    for fp in files:
        base, n, with_ref, agree, mism, used_flowid, used_fallback = check_one(
            fp, args.ref_dir, args.sample, args.seed, args.tolerance_s
        )
        total_rows += n
        total_with_ref += with_ref
        total_mism += mism
        total_flowid += used_flowid
        total_fallback += used_fallback

        print(base)
        print(f"  sample_rows: {n}")
        print(f"  coverage(with_ref): {with_ref} ({(with_ref/n*100 if n else 0):.2f}%)")
        print(f"  label_agreement: {agree:.4f}")
        print(f"  mismatches: {mism}")
        print(f"  matched_flowid: {used_flowid}")
        print(f"  matched_fallback: {used_fallback}\n")

    print("==== SUMMARY ====")
    print(f"total_sample_rows: {total_rows}")
    print(f"total_with_ref: {total_with_ref} ({(total_with_ref/total_rows*100 if total_rows else 0):.2f}%)")
    print(f"total_mismatches: {total_mism}")
    print(f"total_matched_flowid: {total_flowid}")
    print(f"total_matched_fallback: {total_fallback}")

if __name__ == "__main__":
    main()
