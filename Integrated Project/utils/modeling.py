import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from utils.metrics import smape


def train_models(X_train, y_r, y_f, preprocessor):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scorer = lambda est, X, y: -smape(y, est.predict(X))  # currently unused

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    pipe_r = Pipeline([("prep", preprocessor), ("model", rf)])
    pipe_f = Pipeline([("prep", preprocessor), ("model", rf)])

    pipe_r.fit(X_train, y_r)
    pipe_f.fit(X_train, y_f)

    rf_r_smape = smape(y_r, pipe_r.predict(X_train))
    rf_f_smape = smape(y_f, pipe_f.predict(X_train))
    rf_weighted = 0.25 * rf_r_smape + 0.75 * rf_f_smape

    lr_r = Pipeline([("prep", preprocessor), ("model", LinearRegression())])
    lr_f = Pipeline([("prep", preprocessor), ("model", LinearRegression())])

    lr_r.fit(X_train, y_r)
    lr_f.fit(X_train, y_f)

    lr_r_smape = smape(y_r, lr_r.predict(X_train))
    lr_f_smape = smape(y_f, lr_f.predict(X_train))
    lr_weighted = 0.25 * lr_r_smape + 0.75 * lr_f_smape

    if lr_weighted < rf_weighted:
        winner = "LinearRegression"
        best_r = lr_r
        best_f = lr_f
    else:
        winner = "RandomForest"
        best_r = pipe_r
        best_f = pipe_f

    return {
        "winner": winner,
        "best_rougher": best_r,
        "best_final": best_f,
        "comparison": {
            "RandomForest": rf_weighted,
            "LinearRegression": lr_weighted
        }
    }