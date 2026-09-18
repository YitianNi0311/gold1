"""Train and evaluate the local three-tree ensemble using the historical protocol."""

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from no_fcnn_model import preprocess_data, prepare_fold, train_fold, fit_meta_models
from regression_scale_utils import regression_metrics_original_price


ROOT = Path(__file__).resolve().parent
METRIC_LABELS = {"rmse_usd_per_oz": "RMSE (USD/oz)", "mape_percent": "MAPE (%)"}
PROTOCOL_NOTE = (
    "Historical protocol: classification label is (GOLD.diff() > 0); regression label is GOLD.shift(-1). "
    "The meta-model is trained on labels from all five evaluation intervals. "
    "These results include label and meta-evaluation leakage."
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def announce(message):
    print(message, flush=True)
    logging.info(message)


def classification_metrics(true, predicted, probabilities):
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "precision": float(precision_score(true, predicted, zero_division=0)),
        "recall": float(recall_score(true, predicted, zero_division=0)),
        "f1": float(f1_score(true, predicted, zero_division=0)),
        "auc": float(roc_auc_score(true, probabilities)),
    }


def run(data_path, output_dir):
    """Run the unchanged two-pass five-fold stacking procedure for zero FCNNs."""
    data_path, output_dir = Path(data_path).resolve(), Path(output_dir).resolve()
    X, y_class, y_reg = preprocess_data(data_path)
    raw = pd.read_excel(data_path)
    dates = pd.to_datetime(raw.loc[X.index, "Date"])
    folds = list(TimeSeriesSplit(n_splits=5).split(X))
    # A new directory prevents an accidental overwrite of an earlier run.
    output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(42)
    np.random.seed(42)
    log_handler = logging.FileHandler(output_dir / "training_log.txt", encoding="utf-8")
    log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger = logging.getLogger()
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)
    status = {
        "status": "running", "architecture": "LightGBM + XGBoost + Random Forest -> GBM",
        "fcnn_count": 0, "random_state": 42, "n_splits": 5,
        "optuna_trials_per_task_per_fold_per_phase": 50, "optuna_objective_cv": 3,
        "optuna_sampler_seed": None, "data_path": str(data_path),
        "n_samples": len(X), "n_features": X.shape[1],
        "classification_label": "(GOLD.diff() > 0).astype(int)",
        "regression_label": "GOLD.shift(-1)", "evaluation_protocol": PROTOCOL_NOTE,
        "rmse_unit": "USD/oz", "mape_unit": "%",
    }
    write_json(output_dir / "run_manifest.json", status)
    try:
        announce(status["architecture"])
        announce(PROTOCOL_NOTE)
        regression_features, regression_labels = [], []
        classification_features, classification_labels = [], []
        for fold, (train_index, test_index) in enumerate(folds, 1):
            announce(f"OOF fold {fold}/5: training three base learners per task")
            fd = prepare_fold(X, y_class, y_reg, train_index, test_index)
            reg_features, clf_features = train_fold(fd)
            regression_features.append(reg_features)
            regression_labels.extend(fd["yr_test_fit"])
            classification_features.append(clf_features)
            classification_labels.extend(fd["yc_test"].to_numpy())
        meta_reg, meta_clf = fit_meta_models(
            regression_features, regression_labels, classification_features, classification_labels,
        )

        predictions, metric_rows, boundaries = [], [], []
        for fold, (train_index, test_index) in enumerate(folds, 1):
            announce(f"Evaluation fold {fold}/5: retraining three base learners per task")
            fd = prepare_fold(X, y_class, y_reg, train_index, test_index)
            reg_features, clf_features = train_fold(fd)
            pred_fit = meta_reg.predict(reg_features)
            pred_price, rmse, mape = regression_metrics_original_price(
                fd["yr_test"], pred_fit, prediction_scaler=fd["scaler_y"],
            )
            predicted_class = meta_clf.predict(clf_features)
            probability = meta_clf.predict_proba(clf_features)[:, 1]
            frame = pd.DataFrame({
                "model": "No FCNN (three trees -> GBM)", "fold": fold,
                "row_index": X.index[test_index],
                "date": dates.iloc[test_index].dt.strftime("%Y-%m-%d").to_numpy(),
                "y_true_class": fd["yc_test"].to_numpy(), "y_pred_class": predicted_class,
                "y_pred_probability": probability,
                "y_true_price_original": fd["yr_test"].to_numpy(), "y_pred_price_original": pred_price,
                "y_true_price_scaled": fd["yr_test_fit"], "y_pred_price_scaled": pred_fit,
            })
            if not np.isfinite(frame.select_dtypes(include="number")).all().all():
                raise ValueError("Final predictions contain NaN or infinity")
            frame.to_csv(output_dir / f"fold{fold}_predictions.csv", index=False,
                         encoding="utf-8-sig", float_format="%.17g")
            predictions.append(frame)
            write_json(output_dir / f"scaler_y_fold{fold}.json", {
                "mean": fd["scaler_y"].mean_.tolist(), "scale": fd["scaler_y"].scale_.tolist(),
            })
            scores = classification_metrics(fd["yc_test"], predicted_class, probability)
            scores.update(rmse_usd_per_oz=rmse, mape_percent=mape)
            metric_rows.append({"fold": fold, "n_samples": len(test_index), **scores})
            boundaries.append({
                "fold": fold, "n_train": len(train_index), "n_test": len(test_index),
                "train_start": str(dates.iloc[train_index[0]].date()),
                "train_end": str(dates.iloc[train_index[-1]].date()),
                "test_start": str(dates.iloc[test_index[0]].date()),
                "test_end": str(dates.iloc[test_index[-1]].date()),
            })
            announce(f"Fold {fold}: RMSE (USD/oz)={rmse:.6f}, MAPE (%)={mape:.6f}")
            announce("Classification: " + ", ".join(f"{key}={scores[key]:.6f}" for key in
                                                      ("accuracy", "precision", "recall", "f1", "auc")))

        combined = pd.concat(predictions, ignore_index=True)
        combined.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
        fold_metrics = pd.DataFrame(metric_rows)
        fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
        pd.DataFrame(boundaries).to_csv(output_dir / "fold_boundaries.csv", index=False, encoding="utf-8-sig")
        summary = pd.DataFrame([
            {"metric": metric, "label": METRIC_LABELS.get(metric, metric),
             "mean": float(fold_metrics[metric].mean()), "std": float(fold_metrics[metric].std(ddof=0))}
            for metric in scores
        ])
        summary.to_csv(output_dir / "metric_summary.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
        for row in summary.itertuples():
            announce(f"Five-fold {row.label}: {row.mean:.6f} +/- {row.std:.6f}")
        _, pooled_rmse, pooled_mape = regression_metrics_original_price(
            combined.y_true_price_original, combined.y_pred_price_original,
        )
        pooled = classification_metrics(combined.y_true_class, combined.y_pred_class, combined.y_pred_probability)
        pooled.update(rmse_usd_per_oz=pooled_rmse, mape_percent=pooled_mape)
        write_json(output_dir / "pooled_metrics.json", pooled)
        status.update(status="completed", n_predictions=len(combined))
        announce(f"Completed: {output_dir}")
        return output_dir
    except Exception as exc:
        status.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(output_dir / "run_manifest.json", status)
        logger.removeHandler(log_handler)
        log_handler.close()
        logger.setLevel(previous_level)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "GOLD_cleaned.xlsx")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or ROOT / "no_fcnn_results" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run(args.data, output)


if __name__ == "__main__":
    main()
