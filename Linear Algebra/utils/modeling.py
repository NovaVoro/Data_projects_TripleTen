import math
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier


def eval_classifier(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, normalize="all")
    return f1, cm


def rnd_model_predict(P, size, seed=42):
    rng = np.random.default_rng(seed=seed)
    return rng.binomial(n=1, p=P, size=size)


def get_knn(df: pd.DataFrame, feature_names, n: int, k: int, metric: str):
    nbrs = NearestNeighbors(
        n_neighbors=k,
        metric=metric,
    ).fit(df[feature_names].to_numpy())

    nbrs_distances, nbrs_indices = nbrs.kneighbors(
        [df.iloc[n][feature_names]],
        k,
        return_distance=True,
    )

    df_res = pd.concat(
        [
            df.iloc[nbrs_indices[0]],
            pd.DataFrame(
                nbrs_distances.T,
                index=nbrs_indices[0],
                columns=["distance"],
            ),
        ],
        axis=1,
    )

    return df_res


def knn_classification_experiment(df, df_scaled, feature_names, test_size=0.3, random_state=42):
    X = df[feature_names]
    y = df["insurance_benefits_received"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train_scaled, X_test_scaled, _, _ = train_test_split(
        df_scaled[feature_names],
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    results_unscaled = []
    results_scaled = []

    for k in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        f1, cm = eval_classifier(y_test, y_pred)
        results_unscaled.append({"k": k, "f1": f1, "cm": cm})

    for k in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        f1, cm = eval_classifier(y_test, y_pred)
        results_scaled.append({"k": k, "f1": f1, "cm": cm})

    return results_unscaled, results_scaled


class MyLinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        self.weights = np.linalg.inv(X2.T @ X2) @ X2.T @ y

    def predict(self, X):
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        y_pred = X2 @ self.weights
        return y_pred


def eval_regressor(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2


def run_linear_regression(df, df_scaled, feature_names, random_state=12345):
    X = df[feature_names].to_numpy()
    y = df["insurance_benefits"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=random_state,
    )

    lr = MyLinearRegression()
    lr.fit(X_train, y_train)
    y_test_pred = lr.predict(X_test)
    rmse_unscaled, r2_unscaled = eval_regressor(y_test, y_test_pred)

    X_scaled = df_scaled[feature_names].to_numpy()
    y_scaled = df_scaled["insurance_benefits"].to_numpy()

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_scaled,
        y_scaled,
        test_size=0.3,
        random_state=random_state,
    )

    lr_s = MyLinearRegression()
    lr_s.fit(X_train_s, y_train_s)
    y_test_pred_s = lr_s.predict(X_test_s)
    rmse_scaled, r2_scaled = eval_regressor(y_test_s, y_test_pred_s)

    return {
        "unscaled": {
            "weights": lr.weights,
            "rmse": rmse_unscaled,
            "r2": r2_unscaled,
        },
        "scaled": {
            "weights": lr_s.weights,
            "rmse": rmse_scaled,
            "r2": r2_scaled,
        },
    }