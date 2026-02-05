import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at: {path}")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty or corrupted.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while reading CSV: {e}")
    return df