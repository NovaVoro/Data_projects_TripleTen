import numpy as np

def compute_recovery(feed, conc, tail, df):
    F = df[feed].astype(float)
    C = df[conc].astype(float)
    T = df[tail].astype(float)

    num = C * (F - T)
    den = F * (C - T)

    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.where(den != 0, num / den, np.nan)

    return rec

def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    frac = np.where(denom != 0, np.abs(y_true - y_pred) / denom, 0)
    return float(np.nanmean(frac) * 100)