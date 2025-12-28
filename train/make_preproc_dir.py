from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # katalog projektu (IDS)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.preprocess import clean_flow_df


def load_sample_from_csv(csv_path: Path, n_rows: int) -> pd.DataFrame:
    # skipinitialspace ważne, bo CICIDS2017 ma " Destination Port" itd.
    df = pd.read_csv(csv_path, nrows=n_rows, low_memory=False, skipinitialspace=True)
    return df


def main(data_dir: str, out_path: str, rows_per_file: int = 200000) -> None:
    data_dir_p = Path(data_dir)
    files = sorted(data_dir_p.glob("*.csv"))
    if not files:
        print("No data loaded. Check your CSVs and path.")
        return

    X_list = []
    for f in files:
        raw = load_sample_from_csv(f, rows_per_file)
        Xdf = clean_flow_df(raw)
        if Xdf is None or len(Xdf) == 0:
            continue
        X_list.append(Xdf)
        print(f"Loaded sample {Xdf.shape} from {f.name}")

    if not X_list:
        print("No data loaded. Check your CSVs and path.")
        return

    X = pd.concat(X_list, axis=0, ignore_index=True)

    # Lista cech – to jest PRAWDA treningu
    feature_cols = list(X.columns)

    preproc = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])
    preproc.fit(X.values.astype(np.float32))

    out = {
        "preproc": preproc,
        "feature_cols": feature_cols,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, out_path)
    print(f"Saved preproc (+feature_cols) to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows_per_file", type=int, default=200000)
    args = ap.parse_args()
    main(args.data_dir, args.out, args.rows_per_file)
