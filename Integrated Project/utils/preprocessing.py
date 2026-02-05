import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

TARGET_R = "rougher.output.recovery"
TARGET_F = "final.output.recovery"

def prepare_training_data(train_df, test_df):
    feature_cols_train = set(train_df.columns) - {TARGET_R, TARGET_F}
    feature_cols_test = set(test_df.columns)
    missing = feature_cols_train - feature_cols_test

    train_df = train_df.drop(columns=missing, errors="ignore")

    X_train = train_df.drop(columns=[TARGET_R, TARGET_F])
    y_r = train_df[TARGET_R]
    y_f = train_df[TARGET_F]
    X_test = test_df.copy()

    num = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent"))
        ]), cat)
    ])

    preprocessor.fit(X_train)

    return X_train, y_r, y_f, X_test, preprocessor