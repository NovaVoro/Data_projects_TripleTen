import numpy as np
import pandas as pd

def clean_data(df: pd.DataFrame, target_col: str = "Price") -> pd.DataFrame:
    df = df.copy()

    df = df.dropna(subset=[target_col])

    date_cols = ["DateCrawled", "DateCreated", "LastSeen"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    drop_cols = ["NumberOfPictures", "PostalCode"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    if "RegistrationYear" in df.columns:
        df = df[(df["RegistrationYear"] >= 1950) & (df["RegistrationYear"] <= 2025)]

    if "Power" in df.columns:
        df.loc[(df["Power"] <= 20) | (df["Power"] > 1000), "Power"] = np.nan

    if "Mileage" in df.columns:
        df.loc[(df["Mileage"] <= 0) | (df["Mileage"] > 1_000_000), "Mileage"] = np.nan

    df = df[df[target_col] > 0]
    upper = df[target_col].quantile(0.99)
    df[target_col] = np.clip(df[target_col], 0, upper)

    return df