import os
import json
import numpy as np
import mlflow

from metaflow import FlowSpec, step, Parameter, current

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.load_adult import load_adult
from src.data.split import split_train_val_test
from src.features.preprocess import infer_column_types, build_preprocessor, fit_transform_splits
from src.models.base_models import get_base_model
from src.models.calibrate import calibrate_model
from src.models.trust_oof import calibrated_wrapper, generate_oof_calibrated_probs
from src.features.trust_features import make_trust_features, make_trust_labels
from src.evaluation.trust_metrics import trust_auc, coverage_accuracy_curve


class TrustScoreFlow(FlowSpec):
    seed = Parameter("seed", default=42, type=int)
    test_size = Parameter("test_size", default=0.20, type=float)
    val_size = Parameter("val_size", default=0.20, type=float)
    n_bins = Parameter("n_bins", default=15, type=int)
    oof_splits = Parameter("oof_splits", default=5, type=int)
    inner_cv = Parameter("inner_cv", default=3, type=int)

    mlflow_tracking_uri = Parameter("mlflow_tracking_uri", default="file:./mlruns", type=str)
    mlflow_experiment = Parameter("mlflow_experiment", default="TrustScore", type=str)

    @step
    def start(self):
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment)
        mlflow.start_run(run_name=f"{current.flow_name}-{current.run_id}-{current.step_name}-{current.task_id}")

        mlflow.set_tags(
            {
                "metaflow_flow": current.flow_name,
                "metaflow_run_id": str(current.run_id),
                "metaflow_step": current.step_name,
                "metaflow_task": str(current.task_id),
            }
        )

        mlflow.log_params(
            {
                "seed": self.seed,
                "test_size": self.test_size,
                "val_size": self.val_size,
                "n_bins": self.n_bins,
                "oof_splits": self.oof_splits,
                "inner_cv": self.inner_cv,
            }
        )

        x, y, meta = load_adult(return_meta=True)
        self.meta = meta

        x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(
            x, y, test_size=self.test_size, val_size=self.val_size, seed=self.seed
        )

        self.x_train, self.x_val, self.x_test = x_train, x_val, x_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        mlflow.log_metrics(
            {
                "n_rows": float(meta["n_rows"]),
                "n_cols": float(meta["n_cols"]),
                "target_positive_rate": float(meta["target_positive_rate"]),
            }
        )

        self.model_names = ["logreg", "rf"]

        mlflow.end_run()
        self.next(self.preprocess)

    @step
    def preprocess(self):
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment)
        mlflow.start_run(run_name=f"{current.flow_name}-{current.run_id}-{current.step_name}-{current.task_id}")

        mlflow.set_tags(
            {
                "metaflow_flow": current.flow_name,
                "metaflow_run_id": str(current.run_id),
                "metaflow_step": current.step_name,
                "metaflow_task": str(current.task_id),
            }
        )

        cat_cols, num_cols = infer_column_types(self.x_train)
        self.cat_cols, self.num_cols = cat_cols, num_cols

        pre = build_preprocessor(cat_cols, num_cols, scale_numeric=True)
        xt_train, xt_val, xt_test = fit_transform_splits(pre, self.x_train, self.x_val, self.x_test)

        self.pre = pre
        self.xt_train, self.xt_val, self.xt_test = xt_train, xt_val, xt_test

        mlflow.log_params({"n_categorical_cols": len(cat_cols), "n_numeric_cols": len(num_cols)})

        mlflow.end_run()
        self.next(self.train_base, foreach="model_names")

    @step
    def train_base(self):
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment)
        mlflow.start_run(run_name=f"{current.flow_name}-{current.run_id}-{current.step_name}-{current.task_id}")

        mlflow.set_tags(
            {
                "metaflow_flow": current.flow_name,
                "metaflow_run_id": str(current.run_id),
                "metaflow_step": current.step_name,
                "metaflow_task": str(current.task_id),
            }
        )

        name = self.input
        model = get_base_model(name, seed=self.seed)
        model.fit(self.xt_train, self.y_train)

        p_val = model.predict_proba(self.xt_val)[:, 1]
        pr = float(average_precision_score(self.y_val, p_val))
        auc = float(roc_auc_score(self.y_val, p_val))

        self.base_name = name
        self.base_model = model
        self.base_pr_auc = pr
        self.base_roc_auc = auc

        mlflow.log_metrics({f"val_pr_auc__{name}": pr, f"val_roc_auc__{name}": auc})

        mlflow.end_run()
        self.next(self.join_base)

    @step
    def join_base(self, inputs):
        best = sorted(inputs, key=lambda s: s.base_pr_auc, reverse=True)[0]

        self.best_base_name = best.base_name
        self.best_base_model = best.base_model
        self.best_base_pr_auc = best.base_pr_auc
        self.best_base_roc_auc = best.base_roc_auc

       
        ref = inputs[0]
        self.x_train = ref.x_train
        self.x_val = ref.x_val
        self.x_test = ref.x_test
        self.y_train = ref.y_train
        self.y_val = ref.y_val
        self.y_test = ref.y_test

        # now safe to bundle
        self.data_bundle = {
            "x_train": self.x_train,
            "x_val": self.x_val,
            "x_test": self.x_test,
            "y_train": self.y_train,
            "y_val": self.y_val,
            "y_test": self.y_test,
        }

        self.next(self.start_calibration_split)


    @step
    def start_calibration_split(self):
        self.calib_bundle = {
            "x_train": self.data_bundle["x_train"],
            "x_val": self.data_bundle["x_val"],
            "y_val": self.data_bundle["y_val"],
        }
        self.calib_methods = ["sigmoid", "isotonic"]
        self.next(self.calibration, foreach="calib_methods")

    @step
    def calibration(self):
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment)
        mlflow.start_run(run_name=f"{current.flow_name}-{current.run_id}-{current.step_name}-{current.task_id}")

        mlflow.set_tags(
            {
                "metaflow_flow": current.flow_name,
                "metaflow_run_id": str(current.run_id),
                "metaflow_step": current.step_name,
                "metaflow_task": str(current.task_id),
            }
        )

        os.makedirs("artifacts/figures", exist_ok=True)

        method = self.input

        x_train = self.calib_bundle["x_train"]
        x_val = self.calib_bundle["x_val"]
        y_val = self.calib_bundle["y_val"]

        cat_cols, num_cols = infer_column_types(x_train)
        pre = build_preprocessor(cat_cols, num_cols, scale_numeric=True)
        pre.fit(x_train)
        xt_val = pre.transform(x_val)

        cal_model, cal_metrics, fig_path = calibrate_model(
            self.best_base_model, xt_val, y_val, method=method, n_bins=self.n_bins
        )

        self.cal_method = method
        self.cal_metrics = cal_metrics
        self.fig_path = fig_path

        mlflow.log_metrics(
            {
                "val_ece": cal_metrics["ece"],
                "val_brier": cal_metrics["brier"],
                "val_pr_auc_cal": cal_metrics["pr_auc"],
                "val_roc_auc_cal": cal_metrics["roc_auc"],
            }
        )
        mlflow.log_params({"calibration_method": method, "best_base_model": self.best_base_name})

        if fig_path and os.path.exists(fig_path):
            mlflow.log_artifact(fig_path, artifact_path="figures")

        mlflow.end_run()
        self.next(self.join_calibrate)

    @step
    def join_calibrate(self, inputs):
        best = sorted(inputs, key=lambda s: (s.cal_metrics["ece"], s.cal_metrics["brier"]))[0]

        self.best_calib_method = best.cal_method
        self.best_calib_metrics = best.cal_metrics

        ref = inputs[0]
        self.data_bundle = ref.data_bundle
        self.best_base_name = ref.best_base_name
        self.best_base_pr_auc = ref.best_base_pr_auc
        self.best_base_roc_auc = ref.best_base_roc_auc

        self.next(self.train_trust_oof)


    @step
    def train_trust_oof(self):
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment)
        mlflow.start_run(run_name=f"{current.flow_name}-{current.run_id}-{current.step_name}-{current.task_id}")

        mlflow.set_tags(
            {
                "metaflow_flow": current.flow_name,
                "metaflow_run_id": str(current.run_id),
                "metaflow_step": current.step_name,
                "metaflow_task": str(current.task_id),
            }
        )

        x_train = self.data_bundle["x_train"]
        y_train = self.data_bundle["y_train"]

        oof_p = generate_oof_calibrated_probs(
            x_train_df=x_train,
            y_train=y_train,
            base_model=self.best_base_name,
            calib_method=self.best_calib_method,
            n_splits=self.oof_splits,
            seed=self.seed,
        )

        f_train, feat_names = make_trust_features(oof_p)
        y_trust_train = make_trust_labels(y_train, oof_p, threshold=0.5)

        trust_clf = LogisticRegression(max_iter=2000, random_state=self.seed)
        trust_clf.fit(f_train, y_trust_train)

        trust_score_train = trust_clf.predict_proba(f_train)[:, 1]
        auc_train = trust_auc(y_trust_train, trust_score_train)

        self.trust_model = trust_clf

        mlflow.log_params(
            {
                "best_base_model": self.best_base_name,
                "best_calib_method": self.best_calib_method,
                "trust_features": json.dumps(feat_names),
            }
        )
        mlflow.log_metrics(
            {
                "trust_auc_train_oof": float(auc_train),
                "best_base_val_pr_auc": float(self.best_base_pr_auc),
                "best_base_val_roc_auc": float(self.best_base_roc_auc),
                "best_calib_val_ece": float(self.best_calib_metrics["ece"]),
                "best_calib_val_brier": float(self.best_calib_metrics["brier"]),
            }
        )

        mlflow.end_run()
        self.next(self.evaluate_test)

    @step
    def evaluate_test(self):
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment)
        mlflow.start_run(run_name=f"{current.flow_name}-{current.run_id}-{current.step_name}-{current.task_id}")

        mlflow.set_tags(
            {
                "metaflow_flow": current.flow_name,
                "metaflow_run_id": str(current.run_id),
                "metaflow_step": current.step_name,
                "metaflow_task": str(current.task_id),
            }
        )

        x_train = self.data_bundle["x_train"]
        x_test = self.data_bundle["x_test"]
        y_train = self.data_bundle["y_train"]
        y_test = self.data_bundle["y_test"]

        cat_cols, num_cols = infer_column_types(x_train)
        pre_final = build_preprocessor(cat_cols, num_cols, scale_numeric=True)
        pre_final.fit(x_train)

        xt_train_final = pre_final.transform(x_train)
        xt_test_final = pre_final.transform(x_test)

        base_final = get_base_model(self.best_base_name, seed=self.seed)
        cal_final = calibrated_wrapper(base_final, method=self.best_calib_method, inner_cv=self.inner_cv)
        cal_final.fit(xt_train_final, y_train)

        p_test = cal_final.predict_proba(xt_test_final)[:, 1]

        f_test, _ = make_trust_features(p_test)
        y_trust_test = make_trust_labels(y_test, p_test, threshold=0.5)
        trust_score_test = self.trust_model.predict_proba(f_test)[:, 1]

        trust_auc_test = float(trust_auc(y_trust_test, trust_score_test))

        thresholds = np.linspace(0.50, 0.95, 10)
        curve = coverage_accuracy_curve(y_test, p_test, trust_score_test, thresholds)

        os.makedirs("artifacts", exist_ok=True)
        curve_path = os.path.join("artifacts", "coverage_accuracy_test.csv")
        with open(curve_path, "w") as f:
            f.write("threshold,coverage,accuracy_auto,n_auto,n_total\n")
            for row in curve:
                f.write(
                    f"{row['threshold']},{row['coverage']},{row['accuracy_auto']},"
                    f"{row['n_auto']},{row['n_total']}\n"
                )

        mlflow.log_metrics(
            {
                "trust_auc_test": trust_auc_test,
                "test_pr_auc_basecal": float(average_precision_score(y_test, p_test)),
                "test_roc_auc_basecal": float(roc_auc_score(y_test, p_test)),
            }
        )
        mlflow.log_artifact(curve_path, artifact_path="tables")

        mlflow.end_run()
        self.next(self.end)

    @step
    def end(self):
        print("\nPipeline complete")
        print("Best base model:", self.best_base_name)
        print("Best calibration:", self.best_calib_method)


if __name__ == "__main__":
    TrustScoreFlow()