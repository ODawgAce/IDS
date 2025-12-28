from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.preprocess import clean_flow_df

def make_preproc(df: pd.DataFrame, out_path: str) -> None:
    df = clean_flow_df(df)

    # Zależnie od wersji CICIDS2017: label kolumna bywa "Label"
    if "Label" not in df.columns:
        raise ValueError("No Label column found after cleaning. Check your CSV format.")

    X = df.drop(columns=["Label"], errors="ignore").values.astype(np.float32)

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])
    pipe.fit(X)

    joblib.dump(pipe, out_path)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out", default="./artifacts/preproc.joblib")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    make_preproc(df, args.out)
    print("Saved:", args.out)
