import pandas as pd
import os
from functools import lru_cache

DATA_FILES = [
    ("datasets/geo_data_0.csv", "region_0"),
    ("datasets/geo_data_1.csv", "region_1"),
    ("datasets/geo_data_2.csv", "region_2"),
]

@lru_cache(maxsize=1)
def load_raw_region_data():
    """Load all three region CSVs and return a dict of DataFrames."""
    regions = {}
    for path, label in DATA_FILES:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset: {path}")
        df = pd.read_csv(path)
        df["region"] = label
        regions[label] = df
    return regions