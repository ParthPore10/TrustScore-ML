import numpy as np
from sklearn.metrics import roc_auc_score


def trust_auc(y_trust_true, trust_score):
    y = np.asarray(y_trust_true).astype(int)
    s = np.asarray(trust_score).astype(float)
    return float(roc_auc_score(y, s))


def coverage_accuracy_curve(y_true, p_pred, trust_score, thresholds):
    
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred).astype(float)
    trust_score = np.asarray(trust_score).astype(float)

    pred_label = (p_pred >= 0.5).astype(int)
    correct = (pred_label == y_true).astype(int)

    out = []
    n = len(y_true)

    for t in thresholds:
        auto_mask = trust_score >= t
        cov = float(auto_mask.mean())

        if auto_mask.sum() == 0:
            acc_auto = float("nan")
        else:
            acc_auto = float(correct[auto_mask].mean())

        out.append({
            "threshold": float(t),
            "coverage": cov,
            "accuracy_auto": acc_auto,
            "n_auto": int(auto_mask.sum()),
            "n_total": int(n),
        })

    return out
