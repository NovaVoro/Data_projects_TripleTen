import pandas as pd

TRAIN = "data/gold_recovery_train.csv"
TEST = "data/gold_recovery_test.csv"
FULL = "data/gold_recovery_full.csv"


def load_datasets():
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    full = pd.read_csv(FULL)

    for df in (train, test, full):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df.set_index("date", inplace=True)

    return train, test, full