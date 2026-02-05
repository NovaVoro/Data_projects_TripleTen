import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_and_evaluate(df, feature_cols, target_col, test_size=0.25, random_state=42):
    X_train, X_valid, y_train, y_valid = train_test_split(
        df[feature_cols],
        df[target_col],
        test_size=test_size,
        random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = pd.Series(model.predict(X_valid), index=y_valid.index)
    rmse = float(np.sqrt(mean_squared_error(y_valid, y_pred)))

    return {
        "model": model,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "y_pred": y_pred,
        "rmse": rmse,
        "mean_pred": float(y_pred.mean())
    }

def break_even_threshold(budget, revenue_per_unit, num_wells):
    total_required = budget / revenue_per_unit
    return total_required / num_wells