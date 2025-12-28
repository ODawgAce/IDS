from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_cicids2017_csvs(folder: str) -> pd.DataFrame:
    """
    Wczytuje wszystkie CSV z folderu (rekurencyjnie) i scala w jeden DF.
    """
    paths = list(Path(folder).rglob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__source_file__"] = p.name
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        raise RuntimeError("Could not read any CSVs.")
    return pd.concat(dfs, ignore_index=True)
