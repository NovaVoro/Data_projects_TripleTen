import numpy as np
import pandas as pd

def compute_profit(y_true, y_pred, num_select, budget, revenue_per_unit):
    top_idx = np.argsort(-y_pred)[:num_select]
    total_product = y_true.iloc[top_idx].sum()
    revenue = total_product * revenue_per_unit
    return float(revenue - budget)

def bootstrap_profit(
    y_true,
    y_pred,
    n_bootstrap,
    study_size,
    num_select,
    budget,
    revenue_per_unit,
    random_state=42
):
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    profits = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=study_size, replace=True)
        yt = y_true.iloc[idx].reset_index(drop=True)
        yp = y_pred.iloc[idx].reset_index(drop=True)
        profits.append(
            compute_profit(yt, yp, num_select, budget, revenue_per_unit)
        )

    return np.array(profits)