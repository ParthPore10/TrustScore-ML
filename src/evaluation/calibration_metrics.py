import numpy as np
from sklearn.metrics import brier_score_loss


def expected_calibration_error(y_true, p_pred, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred).astype(float)
    p_pred = np.clip(p_pred, 1e-12, 1 - 1e-12)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p_pred)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]

        if i < n_bins - 1:
            mask = (p_pred >= lo) & (p_pred < hi)
        else:
            mask = (p_pred >= lo) & (p_pred <= hi)

        if mask.sum() == 0:
            continue

        acc = y_true[mask].mean()
        conf = p_pred[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)

    return float(ece)


def brier(y_true, p_pred) -> float:
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred).astype(float)
    p_pred = np.clip(p_pred, 1e-12, 1 - 1e-12)
    return float(brier_score_loss(y_true, p_pred))


def reliability_points(y_true, p_pred, n_bins: int = 15):
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred).astype(float)
    p_pred = np.clip(p_pred, 1e-12, 1 - 1e-12)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mean_pred, frac_pos = [], []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (p_pred >= lo) & (p_pred < hi)
        else:
            mask = (p_pred >= lo) & (p_pred <= hi)

        if mask.sum() == 0:
            continue

        mean_pred.append(p_pred[mask].mean())
        frac_pos.append(y_true[mask].mean())

    return np.asarray(mean_pred), np.asarray(frac_pos)