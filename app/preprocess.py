from __future__ import annotations

import numpy as np
import pandas as pd


# Mapowanie nazw kolumn z CICFlowMeter -> CICIDS2017
RUNTIME_TO_TRAIN_COLS = {
    "Dst Port": "Destination Port",
    "Flow Duration": "Flow Duration",

    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",

    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",

    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",

    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",

    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",

    "Flow IAT Mean": "Flow IAT Mean",
    "Flow IAT Std": "Flow IAT Std",
    "Flow IAT Max": "Flow IAT Max",
    "Flow IAT Min": "Flow IAT Min",

    "Fwd IAT Tot": "Fwd IAT Total",
    "Fwd IAT Mean": "Fwd IAT Mean",
    "Fwd IAT Std": "Fwd IAT Std",
    "Fwd IAT Max": "Fwd IAT Max",
    "Fwd IAT Min": "Fwd IAT Min",

    "Bwd IAT Tot": "Bwd IAT Total",
    "Bwd IAT Mean": "Bwd IAT Mean",
    "Bwd IAT Std": "Bwd IAT Std",
    "Bwd IAT Max": "Bwd IAT Max",
    "Bwd IAT Min": "Bwd IAT Min",

    "Fwd PSH Flags": "Fwd PSH Flags",
    "Bwd PSH Flags": "Bwd PSH Flags",
    "Fwd URG Flags": "Fwd URG Flags",
    "Bwd URG Flags": "Bwd URG Flags",

    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",

    "Fwd Pkts/s": "Fwd Packets/s",
    "Bwd Pkts/s": "Bwd Packets/s",

    "Pkt Len Min": "Min Packet Length",
    "Pkt Len Max": "Max Packet Length",
    "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",

    "FIN Flag Cnt": "FIN Flag Count",
    "SYN Flag Cnt": "SYN Flag Count",
    "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count",
    "ACK Flag Cnt": "ACK Flag Count",
    "URG Flag Cnt": "URG Flag Count",
    "ECE Flag Cnt": "ECE Flag Count",

    "Down/Up Ratio": "Down/Up Ratio",
    "Pkt Size Avg": "Average Packet Size",
    "Fwd Seg Size Avg": "Avg Fwd Segment Size",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size",

    # Bulk (CICFlowMeter skraca nazwy)
    "Fwd Byts/b Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Pkts/b Avg": "Fwd Avg Packets/Bulk",
    "Fwd Blk Rate Avg": "Fwd Avg Bulk Rate",
    "Bwd Byts/b Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Pkts/b Avg": "Bwd Avg Packets/Bulk",
    "Bwd Blk Rate Avg": "Bwd Avg Bulk Rate",

    "Subflow Fwd Pkts": "Subflow Fwd Packets",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Subflow Bwd Byts": "Subflow Bwd Bytes",

    "Init Fwd Win Byts": "Init_Win_bytes_forward",
    "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",

    "Active Mean": "Active Mean",
    "Active Std": "Active Std",
    "Active Max": "Active Max",
    "Active Min": "Active Min",

    "Idle Mean": "Idle Mean",
    "Idle Std": "Idle Std",
    "Idle Max": "Idle Max",
    "Idle Min": "Idle Min",
}

DROP_RUNTIME_COLS = [
    "Flow ID", "Src IP", "Src Port", "Dst IP", "Protocol", "Timestamp"
]


def _normalize_columns(cols: list[str]) -> list[str]:
    out = []
    for c in cols:
        c2 = str(c).replace("\ufeff", "").strip()
        c2 = " ".join(c2.split())  # usuń wielokrotne spacje
        out.append(c2)
    return out


def clean_flow_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Czyści DF z flow features tak, żeby nadawał się do ML:
    - normalizuje nazwy,
    - mapuje CICFlowMeter -> CICIDS2017,
    - usuwa kolumny identyfikacyjne (IP/porty/timestamp/flow id),
    - usuwa duplikaty nazw kolumn (CICIDS2017 ma 2x "Fwd Header Length"),
    - usuwa Label (zostaje do y osobno),
    - konwertuje wszystko na float i czyści inf/nan.
    """
    if df is None or len(df) == 0:
        return df

    df = df.copy()
    df.columns = _normalize_columns(list(df.columns))

    # Mapuj runtime -> trening
    df.rename(columns=RUNTIME_TO_TRAIN_COLS, inplace=True)

    # Usuń runtime kolumny ID jeśli są
    for c in DROP_RUNTIME_COLS:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Usuń duplikaty nazw kolumn (kluczowe!)
    df = df.loc[:, ~df.columns.duplicated()]

    # Usuń Label jeśli jest (y robimy osobno)
    if "Label" in df.columns:
        df.drop(columns=["Label"], inplace=True)

    # Konwersja na liczby
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)

    return df


def make_binary_y(label_series: pd.Series) -> np.ndarray:
    """
    CICIDS2017: BENIGN vs reszta ataków.
    """
    s = label_series.astype(str).str.strip()
    s = s.str.replace("\ufeff", "", regex=False)
    s_up = s.str.upper()
    y = (s_up != "BENIGN").astype(np.int32).to_numpy()
    return y


def align_to_feature_cols(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Dopasowuje DF do dokładnej listy cech użytej w treningu.
    - brakujące: dodaje 0
    - nadmiarowe: usuwa
    - kolejność: identyczna
    """
    df = df.copy()
    df.columns = _normalize_columns(list(df.columns))
    df = df.loc[:, ~df.columns.duplicated()]

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    df = df[feature_cols]

    # pewność dtype float
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df