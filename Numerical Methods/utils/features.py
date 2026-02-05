import numpy as np
import pandas as pd

def engineer_features(df: pd.DataFrame, target_col: str = "Price") -> pd.DataFrame:
    df = df.copy()

    if target_col in df.columns:
        df["LogPrice"] = np.log1p(df[target_col])

    ref_year = 2016

    if "DateCreated" in df.columns and df["DateCreated"].notna().any():
        ref_year = df["DateCreated"].dt.year.fillna(ref_year)
    elif "LastSeen" in df.columns and df["LastSeen"].notna().any():
        ref_year = df["LastSeen"].dt.year.fillna(ref_year)

    if "RegistrationYear" in df.columns:
        df["CarAge"] = ref_year - df["RegistrationYear"]
        df.loc[(df["CarAge"] < 0) | (df["CarAge"] > 80), "CarAge"] = np.nan

    return df