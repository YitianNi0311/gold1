"""BFNE-Net without feature engineering, rebuilt from the paper (not the authors' script).

Uses only the 11 raw variables in Table 1 and drops all 37 engineered features from
Table 2. Everything else is unchanged: two FCNNs + three trees -> GBM, same labels,
same five folds. One run takes a long time because each of the 10 training folds
redoes the Optuna search."""

import argparse
import importlib.util
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from historical_alignment import ROOT, aligned_data
from BFNE_Net_without_FCNNs import prepare_fold, fit_meta_models
from regression_scale_utils import regression_metrics_original_price


RAW_FEATURES = (
    "CrudeOil_SpotPrice_BrentUK", "China_CPI_YoY_CurrentMonth", "US_CPI_YoY_NSA",
    "Japan_CPI_YoY", "SaudiArabia_CPI_YoY", "Turkey_CPI_YoY",
    "US_DowJones_IndustrialAverage", "US_SP500_Index", "SpotRate_USD_CNY",
    "Interbank_OpenRate_USD_INR", "SpotRate_Tokyo_9AM_USD_JPY",
)


def audit_features(x):
    if len(RAW_FEATURES) != 11 or len(set(RAW_FEATURES)) != 11:
        raise ValueError("Table 1 raw feature list is incomplete")
    missing = set(RAW_FEATURES) - set(x.columns)
    if missing:
        raise ValueError(f"Missing Table 1 inputs: {sorted(missing)}")
    engineered = [column for column in x.columns if column not in RAW_FEATURES]
    if len(engineered) != 37 or len(x.columns) != 48:
        raise ValueError("Unexpected engineered feature count")
    selected = x.loc[:, list(RAW_FEATURES)]
    if any(column in engineered for column in selected):
        raise ValueError("Engineered input remains in without-FE model")
    return selected, engineered


def load_full_model_source():
    path = ROOT / "BFNE_Net.py"
    spec = importlib.util.spec_from_file_location("full_bfne_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_fold(base, fd, device):
    reg_parts = base.train_base_models_reg(fd["Xr_train"], fd["yr_train_fit"], device)
    reg_models, reg_opt, reg_sch, reg_stop, reg_loss = reg_parts
    reg_loader = DataLoader(base.FinancialRegressionDataset(
        fd["Xr_train"], fd["yr_train_fit"]), batch_size=32, shuffle=True)
    reg_test = DataLoader(base.FinancialRegressionDataset(
        fd["Xr_test"], fd["yr_test_fit"]), batch_size=32, shuffle=False)
    base.train_fcnn_reg(reg_models, reg_opt, reg_sch, reg_stop, reg_loss,
                        reg_loader, device, num_epochs=300)
    for model in reg_models[2:]:
        model.fit(fd["Xr_train"], fd["yr_train_fit"])
    reg_features = base.generate_meta_features_reg(
        *reg_models, reg_test, fd["Xr_test"], device)

    clf_parts = base.train_base_models_clf(fd["Xc_train"], fd["yc_train"], device)
    clf_models, clf_opt, clf_sch, clf_stop, clf_loss = clf_parts
    clf_loader = DataLoader(base.FinancialClassificationDataset(
        fd["Xc_train"], fd["yc_train"]), batch_size=32, shuffle=True)
    clf_test = DataLoader(base.FinancialClassificationDataset(
        fd["Xc_test"], fd["yc_test"]), batch_size=32, shuffle=False)
    base.train_fcnn_clf(clf_models, clf_opt, clf_sch, clf_stop, clf_loss,
                        clf_loader, device, num_epochs=300)
    for model in clf_models[2:]:
        model.fit(fd["Xc_train"], fd["yc_train"])
    clf_features = base.generate_meta_features_clf(
        *clf_models, clf_test, fd["Xc_test"], device)
    if reg_features.shape[1] != 5 or clf_features.shape[1] != 10:
        raise ValueError("The full two-FCNN stacking dimensions changed")
    return reg_features, clf_features


def run(output_dir):
    x_full, y_class, y_price, _, splits, expected = aligned_data()
    x, deleted = audit_features(x_full)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "run_manifest.json"
    manifest = {
        "status": "running", "source": "new paper-defined without-FE reconstruction",
        "author_implementation_recovered": False,
        "full_model_source": str(ROOT / "BFNE_Net.py"), "seed": 42,
        "n_cleaned_rows": len(x), "n_features": x.shape[1],
        "selected_raw_features_table1": list(RAW_FEATURES),
        "deleted_engineered_features_table2": deleted,
        "architecture": "FCNN1 + FCNN2 + LightGBM + XGBoost + Random Forest -> GBM",
        "labels": {"classification": "(GOLD.diff() > 0)",
                   "regression": "GOLD.shift(-1)"},
        "protocol": "one five-fold OOF stage and one five-fold evaluation stage",
        "leakage_note": "Historical classification label and meta-evaluation leakage remain.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    prior_directory = Path.cwd()
    try:
        os.chdir(output_dir)  # logging goes into the new run dir
        base = load_full_model_source()
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        manifest["device"] = str(device)
        reg_features, reg_labels, clf_features, clf_labels = [], [], [], []
        for fold, (train, test) in enumerate(splits, 1):
            print(f"OOF fold {fold}/5", flush=True)
            fd = prepare_fold(x, y_class, y_price, train, test)
            reg_meta, clf_meta = train_fold(base, fd, device)
            reg_features.append(reg_meta)
            reg_labels.extend(fd["yr_test_fit"])
            clf_features.append(clf_meta)
            clf_labels.extend(fd["yc_test"].to_numpy())
        meta_reg, meta_clf = fit_meta_models(
            reg_features, reg_labels, clf_features, clf_labels)
        frames, metrics = [], []
        for fold, (train, test) in enumerate(splits, 1):
            print(f"Evaluation fold {fold}/5", flush=True)
            fd = prepare_fold(x, y_class, y_price, train, test)
            reg_meta, clf_meta = train_fold(base, fd, device)
            predicted_fit = meta_reg.predict(reg_meta)
            predicted_price, rmse, mape = regression_metrics_original_price(
                fd["yr_test"], predicted_fit, prediction_scaler=fd["scaler_y"])
            predicted_class = meta_clf.predict(clf_meta)
            probability = meta_clf.predict_proba(clf_meta)[:, 1]
            truth_class = fd["yc_test"].to_numpy()
            frame = expected.loc[expected.fold.eq(fold)].copy()
            if not np.array_equal(frame.y_true_class.to_numpy(), truth_class):
                raise ValueError("Classification labels do not match historical fold")
            if not np.allclose(frame.y_true_price, fd["yr_test"], rtol=0, atol=1e-8):
                raise ValueError("Regression prices do not match historical fold")
            frame["y_pred_class"] = predicted_class
            frame["y_pred_probability"] = probability
            frame["y_pred_price_original"] = predicted_price
            frame["y_pred_price_scaled"] = predicted_fit
            frames.append(frame)
            metrics.append({
                "fold": fold, "rmse_usd_per_oz": rmse, "mape_percent": mape,
                "accuracy": accuracy_score(truth_class, predicted_class),
                "precision": precision_score(truth_class, predicted_class, zero_division=0),
                "recall": recall_score(truth_class, predicted_class, zero_division=0),
                "f1": f1_score(truth_class, predicted_class, zero_division=0),
                "auc": roc_auc_score(truth_class, probability),
            })
            frame.to_csv(output_dir / f"fold{fold}_predictions.csv", index=False,
                         encoding="utf-8-sig", float_format="%.17g")
            (output_dir / f"scaler_y_fold{fold}.json").write_text(json.dumps({
                "mean": fd["scaler_y"].mean_.tolist(),
                "scale": fd["scaler_y"].scale_.tolist()}), encoding="utf-8")
        pd.concat(frames, ignore_index=True).to_csv(
            output_dir / "predictions.csv", index=False,
            encoding="utf-8-sig", float_format="%.17g")
        pd.DataFrame(metrics).to_csv(output_dir / "fold_metrics.csv", index=False,
                                     encoding="utf-8-sig", float_format="%.17g")
        manifest["status"] = "completed"
        manifest["n_test_predictions"] = sum(len(frame) for frame in frames)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        raise
    finally:
        os.chdir(prior_directory)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reconstructed_runs" /
                        f"without_fe_{datetime.now():%Y%m%d_%H%M%S}")
    args = parser.parse_args()
    print(run(args.output))
