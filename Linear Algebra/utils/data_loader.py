import pandas as pd
from sklearn.preprocessing import MaxAbsScaler


def load_data(path: str = "data/insurance_us.csv"):
    df = pd.read_csv(path)

    df = df.rename(
        columns={
            "Gender": "gender",
            "Age": "age",
            "Salary": "income",
            "Family members": "family_members",
            "Insurance benefits": "insurance_benefits",
        }
    )

    df["age"] = df["age"].astype(int)

    df["insurance_benefits_received"] = (df["insurance_benefits"] > 0).astype(int)

    df["income_per_member"] = df["income"] / (df["family_members"] + 1)

    feature_names = ["gender", "age", "income", "family_members"]

    scaler = MaxAbsScaler().fit(df[feature_names].to_numpy())
    df_scaled = df.copy()
    df_scaled.loc[:, feature_names] = scaler.transform(df[feature_names].to_numpy())

    return df, df_scaled, feature_names