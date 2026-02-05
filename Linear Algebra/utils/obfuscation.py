import numpy as np
from .modeling import MyLinearRegression, eval_regressor


def generate_random_invertible_matrix(dim: int, seed: int = 42):
    rng = np.random.default_rng(seed=seed)
    P = rng.random(size=(dim, dim))
    det_P = np.linalg.det(P)
    cond_number = np.linalg.cond(P)
    invertible = det_P != 0
    return P, det_P, cond_number, invertible


def transform_features(X, P):
    return X @ P


def recover_features(X_prime, P):
    P_inv = np.linalg.inv(P)
    return X_prime @ P_inv


def test_lr_with_obfuscation(X, y, P):
    lr_orig = MyLinearRegression()
    lr_orig.fit(X, y)
    y_pred_orig = lr_orig.predict(X)
    rmse_orig, r2_orig = eval_regressor(y, y_pred_orig)

    X_prime = transform_features(X, P)
    lr_obf = MyLinearRegression()
    lr_obf.fit(X_prime, y)
    y_pred_obf = lr_obf.predict(X_prime)
    rmse_obf, r2_obf = eval_regressor(y, y_pred_obf)

    return {
        "original": {"rmse": rmse_orig, "r2": r2_orig},
        "obfuscated": {"rmse": rmse_obf, "r2": r2_obf},
    }