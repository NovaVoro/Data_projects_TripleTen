import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

TARGET_R = "rougher.output.recovery"
TARGET_F = "final.output.recovery"


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_features)
    ])


def prepare_training_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    # 1. Drop rows where either target is missing
    df = train_df.dropna(subset=[TARGET_R, TARGET_F])

    # 2. Extract targets
    y_r = df[TARGET_R]
    y_f = df[TARGET_F]

    # 3. Remove ALL output and calculation columns from training
    cols_to_drop = [
        c for c in df.columns
        if ".output." in c or ".calculation." in c or c in [TARGET_R, TARGET_F]
    ]

    X_train_full = df.drop(columns=cols_to_drop)

    # 4. Clean test_df the same way (defensive)
    X_test_full = test_df.drop(
        columns=[c for c in test_df.columns if ".output." in c or ".calculation." in c],
        errors="ignore"
    )

    # 5. Align train/test feature columns
    common_cols = sorted(set(X_train_full.columns) & set(X_test_full.columns))

    X_train = X_train_full[common_cols].copy()
    X_test = X_test_full[common_cols].copy()

    # 6. Build preprocessor on aligned training columns
    preprocessor = build_preprocessor(X_train)

    return X_train, y_r, y_f, X_test, preprocessor