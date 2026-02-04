# model_core.py
import os
import traceback
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# -----------------------------
# SAFE CSV LOADING
# -----------------------------
def safe_read_csv(path: str) -> pd.DataFrame:
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found at path: {path}")
        df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        traceback.print_exc()
        raise


# -----------------------------
# TIME FEATURES
# -----------------------------
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    df = df.copy()
    dt = df.index
    df["hour"] = dt.hour
    df["dayofweek"] = dt.dayofweek
    df["day"] = dt.day
    df["month"] = dt.month
    return df


# -----------------------------
# LAG & ROLLING FEATURES
# -----------------------------
def create_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: List[int],
    rolling_windows: List[int],
) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    for w in rolling_windows:
        df[f"rolling_mean_{w}"] = (
            df[target_col]
            .shift(1)
            .rolling(window=w, min_periods=1)
            .mean()
        )
    return df


# -----------------------------
# TRAIN/TEST SPLIT
# -----------------------------
def train_test_split_time_series(
    df: pd.DataFrame,
    target_col: str,
    test_size_ratio: float = 0.10,
):
    if not 0 < test_size_ratio < 1:
        raise ValueError("test_size_ratio must be between 0 and 1.")
    df = df.sort_index()
    n_samples = len(df)
    test_size = int(np.floor(n_samples * test_size_ratio))
    if test_size == 0:
        raise ValueError("Test size is zero; increase test_size_ratio.")
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    return X_train, X_test, y_train, y_test


# -----------------------------
# MODEL EVALUATION (LOG SPACE)
# -----------------------------
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="model"):
    try:
        model.fit(X_train, y_train)

        # Predict in log-space
        y_pred_train_log = model.predict(X_train)
        y_pred_test_log = model.predict(X_test)

        # Invert transform
        y_pred_train = np.expm1(y_pred_train_log)
        y_pred_test = np.expm1(y_pred_test_log)

        # Invert true values
        y_train_orig = np.expm1(y_train)
        y_test_orig = np.expm1(y_test)

        # RMSE (manual, compatible with all sklearn versions)
        train_rmse = mean_squared_error(y_train_orig, y_pred_train) ** 0.5
        test_rmse = mean_squared_error(y_test_orig, y_pred_test) ** 0.5

        print(f"\n[{model_name}]")
        print(f"  Train RMSE: {train_rmse:.3f}")
        print(f"  Test  RMSE: {test_rmse:.3f}")

        return {
            "model": model,
            "model_name": model_name,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
        }

    except Exception as e:
        print(f"[ERROR] Failed to evaluate model '{model_name}': {e}")
        traceback.print_exc()
        raise


# -----------------------------
# FULL PIPELINE: LOAD → FEATS → TRAIN
# -----------------------------
def build_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = safe_read_csv(csv_path)
    raw_target = "num_orders"

    # Resample to hourly
    df_hourly = raw.resample("1H").sum()
    df_hourly[raw_target] = df_hourly[raw_target].fillna(0)

    # Log transform
    df_hourly["log_orders"] = np.log1p(df_hourly[raw_target])
    target_col = "log_orders"

    # Features
    df_features = create_time_features(df_hourly)
    df_features = create_lag_features(
        df_features,
        target_col=target_col,
        lags=[1, 2, 3, 24, 48],
        rolling_windows=[3, 6, 12, 24],
    )

    df_features = df_features.dropna()

    # Remove raw + target to avoid leakage, then add back only target
    df_features = df_features.drop(columns=["num_orders", "log_orders"], errors="ignore")
    df_features[target_col] = df_hourly[target_col]

    return df_features, df_hourly


def train_all_models(csv_path: str) -> Dict[str, Any]:
    df_features, df_hourly = build_dataset(csv_path)
    target_col = "log_orders"

    X_train, X_test, y_train, y_test = train_test_split_time_series(
        df_features,
        target_col=target_col,
        test_size_ratio=0.10,
    )

    # Linear Regression
    lr_model = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LinearRegression()),
    ])
    lr_results = evaluate_model(lr_model, X_train, y_train, X_test, y_test, "Linear Regression")

    # Random Forest
    rf = RandomForestRegressor(random_state=42)
    rf_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    }
    tscv = TimeSeriesSplit(n_splits=3)
    rf_search = GridSearchCV(
        rf,
        rf_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    rf_results = evaluate_model(best_rf, X_train, y_train, X_test, y_test, "Random Forest")

    # Gradient Boosting
    gb = GradientBoostingRegressor(random_state=42)
    gb_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    }
    gb_search = GridSearchCV(
        gb,
        gb_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    gb_search.fit(X_train, y_train)
    best_gb = gb_search.best_estimator_
    gb_results = evaluate_model(best_gb, X_train, y_train, X_test, y_test, "Gradient Boosting")

    results = {
        "Linear Regression": lr_results,
        "Random Forest": rf_results,
        "Gradient Boosting": gb_results,
    }

    # Pick best by test RMSE
    best_name = min(results, key=lambda k: results[k]["test_rmse"])
    best = results[best_name]

    return {
        "best_model_name": best_name,
        "best_model": best["model"],
        "best_test_rmse": best["test_rmse"],
        "all_results": results,
        "df_features": df_features,
        "df_hourly": df_hourly,
        "target_col": "log_orders",
    }


def predict_next_hour(model, df_features: pd.DataFrame, df_hourly: pd.DataFrame) -> Dict[str, Any]:
    """
    Use the last row of df_features to predict the next hour.
    """
    target_col = "log_orders"

    # Last feature row
    last_row = df_features.drop(columns=[target_col]).iloc[[-1]]

    # Predict log-orders, then invert
    y_pred_log = model.predict(last_row)[0]
    y_pred = np.expm1(y_pred_log)

    # For context, also return last actual hour
    last_time = df_features.index[-1]
    next_time = last_time + pd.Timedelta(hours=1)

    return {
        "last_time": str(last_time),
        "next_time": str(next_time),
        "prediction": float(y_pred),
    }