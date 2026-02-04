import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv("games.csv")

    df.columns = df.columns.str.lower()

    df["user_score"] = df["user_score"].replace("tbd", np.nan)
    df["user_score"] = df["user_score"].astype(float)

    df["year_of_release"] = df["year_of_release"].replace([np.inf, -np.inf], np.nan)
    df["year_of_release"] = df["year_of_release"].fillna(-1).astype(int)
    df = df[df["year_of_release"] != -1]

    df["total_sales"] = (
        df["na_sales"] + df["eu_sales"] + df["jp_sales"] + df["other_sales"]
    )

    return df