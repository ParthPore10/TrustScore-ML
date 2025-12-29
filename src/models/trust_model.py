import os
import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score

from src.evaluation.calibration_metrics import expected_calibration_error, brier, reliability_points
from src.utils.plotting import plot_reliability


def calibrate_model(base_model, Xt_val, y_val, method="sigmoid", n_bins=15):
    method = method.lower().strip()
    if method not in ["sigmoid", "isotonic"]:
        raise ValueError("method must be 'sigmoid' or 'isotonic'")

    # ---- HARD sanity checks (this will catch your current issue immediately) ----
    if isinstance(base_model, str):
        raise TypeError(f"base_model is a string ({base_model}). Pass the fitted estimator object, not the name.")
    if not hasattr(base_model, "fit"):
        raise TypeError("base_model does not implement fit()")
    if not (hasattr(base_model, "predict_proba") or hasattr(base_model, "decision_function")):
        raise TypeError("base_model must implement predict_proba() or decision_function()")

    # ---- scikit-learn API compatibility (estimator vs base_estimator) ----
    try:
        cal = CalibratedClassifierCV(estimator=base_model, method=method, cv="prefit")
    except TypeError:
        cal = CalibratedClassifierCV(base_estimator=base_model, method=method, cv="prefit")

    cal.fit(Xt_val, y_val)

    p_val = cal.predict_proba(Xt_val)[:, 1]

    metrics = {
        "ece": expected_calibration_error(y_val, p_val, n_bins=n_bins),
        "brier": brier(y_val, p_val),
        "roc_auc": float(roc_auc_score(y_val, p_val)),
        "pr_auc": float(average_precision_score(y_val, p_val)),
        "method": method,
    }

    mean_pred, frac_pos = reliability_points(y_val, p_val, n_bins=n_bins)
    fig_path = os.path.join("artifacts", "figures", f"reliability_{method}.png")
    plot_reliability(mean_pred, frac_pos, fig_path, f"Reliability Diagram ({method})")

    return cal, metrics, fig_path


def pick_best_calibration(calib_results):
    return min(calib_results, key=lambda r: (r["metrics"]["ece"], r["metrics"]["brier"]))