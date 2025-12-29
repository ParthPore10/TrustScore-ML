import os
import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score,average_precision_score

from src.evaluation.calibration_metrics import expected_calibration_error,brier,reliability_points
from src.utils.plotting import plot_reliability

def calibrate_model(base_model,
                    xt_val,y_val,
                    method:str ="sigmod",
                    n_bins: int=15):
    
    method = method.lower().strip()

    if method not in ("sigmoid","isotonic"):
        raise ValueError("method must be 'sigmoid' or 'isotonic' ")
    
    cal = CalibratedClassifierCV(estimator=base_model,method=method,cv="prefit")
    cal.fit(xt_val,y_val)

    p_val = cal.predict_proba(xt_val)[:,1]
    
    metrics = {
        "ece": expected_calibration_error(y_val, p_val, n_bins=n_bins),
        "brier": brier(y_val, p_val),
        "roc_auc": float(roc_auc_score(y_val, p_val)),
        "pr_auc": float(average_precision_score(y_val, p_val)),
        "method": method,
    }

    mean_pred,frac_pos = reliability_points(y_val,p_val,n_bins=n_bins)

    mean_pred, frac_pos = reliability_points(y_val, p_val, n_bins=n_bins)
    fig_path = os.path.join("artifacts", "figures", f"reliability_{method}.png")
    plot_reliability(mean_pred, frac_pos, fig_path, f"Reliability Diagram ({method})")

    return cal, metrics, fig_path


def pick_best_calibration(calib_results):
    """
    calib_results: list of dicts, each dict has keys: metrics, model, fig_path
    Selection: lowest ECE, tie-breaker lowest Brier.
    """
    return min(calib_results, key=lambda r: (r["metrics"]["ece"], r["metrics"]["brier"]))


if __name__ == "__main__":
    # Smoke test: train base model -> calibrate sigmoid/isotonic -> compare
    from src.data.load_adult import load_adult
    from src.data.split import split_train_val_test
    from src.features.preprocess import infer_column_types, build_preprocessor, fit_transform_splits
    from src.models.base_models import train_and_eval

    x, y, meta = load_adult(return_meta=True)
    x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(X, y)

    cat_cols, num_cols = infer_column_types(x_train)
    pre = build_preprocessor(cat_cols, num_cols, scale_numeric=True)
    xt_train, xt_val, xt_test = fit_transform_splits(pre, x_train, x_val, x_test)

    # Train two base models, choose best by PR-AUC on val
    base_results = []
    for name in ["logreg", "rf"]:
        model, metrics, _ = train_and_eval(name, xt_train, y_train, xt_val, y_val, seed=42)
        base_results.append((name, model, metrics))

    best_name, best_model, best_metrics = sorted(base_results, key=lambda t: t[2]["pr_auc"], reverse=True)[0]
    print(f"Best base model by PR-AUC: {best_name}  (PR-AUC={best_metrics['pr_auc']:.4f})")

    # Calibrate best model
    results = []
    for method in ["sigmoid", "isotonic"]:
        cal_model, cal_metrics, fig_path = calibrate_model(best_model, xt_val, y_val, method=method, n_bins=15)
        results.append({"model": cal_model, "metrics": cal_metrics, "fig_path": fig_path})

    print("\nCalibration comparison (on VAL):")
    print("Method     ECE       Brier     ROC-AUC   PR-AUC   Plot")
    print("-----------------------------------------------------------")
    for r in sorted(results, key=lambda d: (d["metrics"]["ece"], d["metrics"]["brier"])):
        m = r["metrics"]
        print(f"{m['method']:<10} {m['ece']:<8.4f} {m['brier']:<9.4f} {m['roc_auc']:<8.4f} {m['pr_auc']:<7.4f} {r['fig_path']}")

    best = pick_best_calibration(results)
    print(f"\nSelected calibration: {best['metrics']['method']} (lowest ECE, then Brier)")