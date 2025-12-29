import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.load_adult import load_adult
from src.data.split import split_train_val_test
from src.features.preprocess import infer_column_types, build_preprocessor, fit_transform_splits
from src.models.base_models import get_base_model
from src.features.trust_features import make_trust_features, make_trust_labels
from src.evaluation.trust_metrics import trust_auc, coverage_accuracy_curve
from src.models.calibrate import pick_best_calibration, calibrate_model


def calibrated_wrapper(estimator, method: str, inner_cv: int = 3):
    try:
        return CalibratedClassifierCV(estimator=estimator, method=method, cv=inner_cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method=method, cv=inner_cv)


def generate_oof_calibrated_probs(
    x_train_df,
    y_train,
    base_model: str,
    calib_method: str = "sigmoid",
    n_splits: int = 5,
    seed: int = 42,
):
    y_train = np.asarray(y_train).astype(int)
    oof_p = np.zeros(len(y_train), dtype=float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train_df, y_train), start=1):
        x_tr = x_train_df.iloc[tr_idx]
        y_tr = y_train[tr_idx]
        x_va = x_train_df.iloc[va_idx]

        cat_cols, num_cols = infer_column_types(x_tr)
        pre = build_preprocessor(cat_cols, num_cols, scale_numeric=True)
        pre.fit(x_tr)

        xt_tr = pre.transform(x_tr)
        xt_va = pre.transform(x_va)

        base = get_base_model(base_model, seed=seed)
        cal = calibrated_wrapper(base, method=calib_method, inner_cv=3)
        cal.fit(xt_tr, y_tr)

        oof_p[va_idx] = cal.predict_proba(xt_va)[:, 1]

        print(f"[OOF] fold={fold}/{n_splits} done. val_size={len(va_idx)}")

    return oof_p


def train_trust_model(f_train, y_trust_train, seed: int = 42):
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(f_train, y_trust_train)
    return clf


def predict_trust_score(trust_model, f):
    return trust_model.predict_proba(f)[:, 1]


if __name__ == "__main__":
    x, y, meta = load_adult(return_meta=True)
    x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(x, y)

    cat_cols, num_cols = infer_column_types(x_train)
    pre = build_preprocessor(cat_cols, num_cols, scale_numeric=True)
    xt_train, xt_val, xt_test = fit_transform_splits(pre, x_train, x_val, x_test)

    candidates = []
    for name in ["logreg", "rf"]:
        model = get_base_model(name, seed=42)
        model.fit(xt_train, y_train)
        p_val = model.predict_proba(xt_val)[:, 1]

        candidates.append(
            {
                "name": name,
                "model": model,
                "pr_auc": float(average_precision_score(y_val, p_val)),
                "roc_auc": float(roc_auc_score(y_val, p_val)),
            }
        )

    best_base = sorted(candidates, key=lambda d: d["pr_auc"], reverse=True)[0]
    best_name = best_base["name"]
    print(
        f"\nBest base by VAL PR-AUC: {best_name} "
        f"(PR-AUC={best_base['pr_auc']:.4f}, ROC-AUC={best_base['roc_auc']:.4f})"
    )

    calibs = []
    for method in ["sigmoid", "isotonic"]:
        cal_model, cal_metrics, fig_path = calibrate_model(
            best_base["model"], xt_val, y_val, method=method, n_bins=15
        )
        calibs.append({"model": cal_model, "metrics": cal_metrics, "fig_path": fig_path})

    best_cal = pick_best_calibration(calibs)
    chosen_method = best_cal["metrics"]["method"]
    print(
        f"Chosen calibration on VAL: {chosen_method} "
        f"(ECE={best_cal['metrics']['ece']:.4f}, Brier={best_cal['metrics']['brier']:.4f})"
    )

    oof_p_train = generate_oof_calibrated_probs(
        x_train_df=x_train,
        y_train=y_train,
        base_model=best_name,
        calib_method=chosen_method,
        n_splits=5,
        seed=42,
    )

    f_trust_train, feat_names = make_trust_features(oof_p_train)
    y_trust_train = make_trust_labels(y_train, oof_p_train, threshold=0.5)

    trust_clf = train_trust_model(f_trust_train, y_trust_train, seed=42)
    trust_score_train = predict_trust_score(trust_clf, f_trust_train)

    print("\nTrust model (trained on TRAIN OOF correctness):")
    print("  trust_auc_train(oof):", round(trust_auc(y_trust_train, trust_score_train), 4))

    cat_cols, num_cols = infer_column_types(x_train)
    pre_final = build_preprocessor(cat_cols, num_cols, scale_numeric=True)
    pre_final.fit(x_train)
    xt_train_final = pre_final.transform(x_train)

    base_final = get_base_model(best_name, seed=42)
    cal_final = calibrated_wrapper(base_final, method=chosen_method, inner_cv=3)
    cal_final.fit(xt_train_final, y_train)

    xt_test_final = pre_final.transform(x_test)
    p_test_cal = cal_final.predict_proba(xt_test_final)[:, 1]

    f_test, _ = make_trust_features(p_test_cal)
    y_trust_test = make_trust_labels(y_test, p_test_cal, threshold=0.5)
    trust_score_test = predict_trust_score(trust_clf, f_test)

    print("  trust_auc_test:", round(trust_auc(y_trust_test, trust_score_test), 4))

    thresholds = np.linspace(0.5, 0.95, 10)
    curve = coverage_accuracy_curve(y_test, p_test_cal, trust_score_test, thresholds)

    print("\nCoverage vs Accuracy (TEST):")
    print("thr     coverage   acc_auto   n_auto")
    print("------------------------------------")
    for row in curve:
        thr = row["threshold"]
        cov = row["coverage"]
        acc = row["accuracy_auto"]
        n_auto = row["n_auto"]
        acc_str = "nan" if np.isnan(acc) else f"{acc:.4f}"
        print(f"{thr:.2f}    {cov:.4f}     {acc_str:<7}   {n_auto}")