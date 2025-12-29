import numpy as np

def binary_entropy(p):
    p = np.asarray(p,dtype=float)
    p = np.clip(p,1e-12,1-1e-12)
    return -(p*np.log(p)+(1-p)*np.log(1-p))


def make_trust_features(p_cal):

    p = np.asarray(p_cal,dtype=float)
    p = np.clip(p,1e-12,1 - 1e-12)

    margin = np.abs(p-0.5)
    ent = binary_entropy(p)
    pred = (p>=0.5).astype(int)

    f = np.column_stack([p, margin, ent, pred])
    feature_names = ["p_cal", "margin_abs", "entropy", "pred_label"]
    return f, feature_names


def make_trust_labels(y_true, p_cal, threshold: float = 0.5):
    
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p_cal, dtype=float)
    pred = (p >= threshold).astype(int)
    return (pred == y_true).astype(int)
