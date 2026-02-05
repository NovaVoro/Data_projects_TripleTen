import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

def build_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=12,
            min_samples_leaf=5,
            random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            n_jobs=-1,
            random_state=42
        )
    }

    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            random_state=42,
            n_jobs=-1
        )

    return models

def train_and_evaluate(name, model, preprocessor, X, y, silent=False):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_valid)

    mse = mean_squared_error(
        np.expm1(y_valid),
        np.expm1(preds)
    )
    rmse = np.sqrt(mse)


    if not silent:
        print(f"{name} RMSE: {rmse:.2f}")

    return pipe, rmse