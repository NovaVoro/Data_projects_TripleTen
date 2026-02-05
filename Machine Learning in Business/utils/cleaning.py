import pandas as pd
import numpy as np

def deduplicate_file(df: pd.DataFrame, region_label: str) -> pd.DataFrame:
    df_dedup = df.drop_duplicates(subset="id", keep="first").copy()
    df_dedup["region"] = region_label
    return df_dedup

def review_duplicates(df: pd.DataFrame):
    within = df[df.duplicated(subset=["id", "region"], keep=False)]
    cross = (
        df.groupby("id")["region"]
        .nunique()
        .reset_index()
        .query("region > 1")
    )
    return within, cross

def cap_outliers(series: pd.Series) -> pd.Series:
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series.clip(lower=lower, upper=upper)

def apply_outlier_capping(df: pd.DataFrame, feature_cols):
    df = df.copy()
    for col in feature_cols:
        df[col] = cap_outliers(df[col])
    return df