"""Read-only audit of retained predictions; writes new audit artifacts, never trains."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import ta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from regression_scale_utils import regression_metrics_original_price


ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_preprocessing(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "preprocess_data")
    namespace = {"np": np, "pd": pd, "ta": ta}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["preprocess_data"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True,
                        help="Retained run directory containing full/ and no_fcnn/ price CSVs")
    parser.add_argument("--table-root", type=Path, required=True,
                        help="Retained tree prediction CSVs and metric summaries")
    parser.add_argument("--output", type=Path, default=ROOT / "历史状态复核_20260917")
    args = parser.parse_args()
    reference = args.reference_root.resolve()
    history = args.prediction_root.resolve()
    table = args.table_root.resolve()
    output = args.output.resolve()
    if any(output == path or path in output.parents for path in (reference, history, table)):
        raise ValueError("Audit output must be outside the preserved historical project")
    output.mkdir(parents=True, exist_ok=True)
    evidence = {}

    def read_csv(path):
        evidence[str(path)] = sha(path)
        return pd.read_csv(path, float_precision="round_trip")

    for name in ("归一.py", "RF.py", "XGBoost.py", "gold lightgbm.py", "GOLD_cleaned.xlsx"):
        assert sha(ROOT / name) == sha(reference / name), f"Reference mismatch: {name}"
        evidence[str(reference / name)] = sha(reference / name)

    # Execute only the unchanged feature/label function, never the training module.
    X, yc, yr = load_preprocessing(ROOT / "归一.py")(ROOT / "GOLD_cleaned.xlsx")
    raw = pd.read_excel(ROOT / "GOLD_cleaned.xlsx")
    np.testing.assert_array_equal(yc, (raw.GOLD.diff() > 0).astype(int).loc[X.index])
    np.testing.assert_array_equal(yr, raw.GOLD.shift(-1).loc[X.index])
    current_gold = 3 * X.GOLD_MA_3_days - X.GOLD_lag1 - X.GOLD_lag2
    np.testing.assert_allclose(current_gold, raw.GOLD.loc[X.index], atol=1e-9, rtol=0)
    folds = list(TimeSeriesSplit(n_splits=5).split(X))
    test_indices = np.concatenate([test for _, test in folds])
    true_prices = yr.iloc[test_indices].to_numpy()
    expected_dates = pd.to_datetime(raw.Date.loc[X.index].iloc[test_indices]).dt.strftime("%Y-%m-%d").to_numpy()
    assert len(test_indices) == 3680 and all(len(test) == 736 for _, test in folds)

    rows = []
    pooled = []
    scaler_checks = []
    serialization_checks = []

    def score(model, frame):
        np.testing.assert_array_equal(frame.y_true_price, true_prices)
        for fold, group in frame.groupby("fold", sort=True):
            _, rmse, mape = regression_metrics_original_price(group.y_true_price, group.y_pred_price_original)
            rows.append({"model": model, "fold": int(fold), "n": len(group),
                         "rmse_usd_per_oz": rmse, "mape_percent": mape})
        _, rmse, mape = regression_metrics_original_price(frame.y_true_price, frame.y_pred_price_original)
        pooled.append({"model": model, "n": len(frame), "rmse_usd_per_oz": rmse, "mape_percent": mape})

    for key, model in (("full", "BFNE-Net"), ("no_fcnn", "BFNE-Net without FCNNs")):
        frame = read_csv(history / key / "predictions.csv")
        np.testing.assert_array_equal(frame.date, expected_dates)
        np.testing.assert_array_equal(frame.y_true_class, yc.iloc[test_indices])
        np.testing.assert_array_equal(frame.fold, np.repeat(np.arange(1, 6), 736))
        for fold, (train, test) in enumerate(folds, 1):
            scaler = StandardScaler().fit(yr.iloc[train].to_numpy().reshape(-1, 1))
            path = history / key / f"scaler_y_fold{fold}.json"
            evidence[str(path)] = sha(path)
            retained = json.loads(path.read_text(encoding="utf-8"))
            np.testing.assert_allclose(scaler.mean_, retained["mean"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(scaler.scale_, retained["scale"], atol=1e-12, rtol=0)
            group = frame[frame.fold == fold]
            np.testing.assert_array_equal(group.y_true_price, yr.iloc[test])
            predicted, rmse, mape = regression_metrics_original_price(
                group.y_true_price, group.y_pred_price_scaled, prediction_scaler=scaler)
            np.testing.assert_allclose(predicted, group.y_pred_price_original, atol=1e-9, rtol=0)
            np.testing.assert_allclose(scaler.transform(group.y_true_price.to_numpy().reshape(-1, 1)).ravel(),
                                       group.y_true_price_scaled, atol=1e-12, rtol=0)
            scaler_checks.append({"model": model, "fold": fold, "mean": float(scaler.mean_[0]),
                                  "scale": float(scaler.scale_[0]), "rmse_usd_per_oz": rmse,
                                  "mape_percent": mape, "training_fold_scaler_matches": True,
                                  "max_price_reconstruction_error": float(np.max(np.abs(predicted - group.y_pred_price_original)))})
        score(model, frame)

    for prefix, model, filename in (("RF", "Random Forest", "RF.py"),
                                    ("XGBoost", "XGBoost", "XGBoost.py"),
                                    ("LightGBM", "LightGBM", "gold lightgbm.py")):
        frame = read_csv(table / f"{prefix}_regression_predictions_original_price.csv")
        frame = frame.rename(columns={"y_true_price_original": "y_true_price"})
        np.testing.assert_array_equal(frame.row_index, X.index[test_indices])
        np.testing.assert_array_equal(frame.fold, np.repeat(np.arange(1, 6), 736))
        tree_x, tree_yc, tree_yr = load_preprocessing(ROOT / filename)(ROOT / "GOLD_cleaned.xlsx")
        pd.testing.assert_frame_equal(tree_x, X)
        np.testing.assert_array_equal(tree_yc, yc)
        np.testing.assert_array_equal(tree_yr, yr)
        score(model, frame)
        saved = read_csv(table / f"{prefix}_metric_summary.csv").set_index("metric")
        calculated = pd.DataFrame([row for row in rows if row["model"] == model])
        logged_calculated = calculated.copy()
        if prefix == "XGBoost":
            # pandas wrote the native float32 predictions as short decimal strings.
            # Recover that dtype only to validate the metrics computed before CSV
            # serialization; the audit output itself scores the saved CSV values.
            native_rows = []
            for fold, group in frame.groupby("fold", sort=True):
                native_predictions = group.y_pred_price_original.to_numpy(dtype=np.float32).astype(float)
                _, rmse, mape = regression_metrics_original_price(group.y_true_price, native_predictions)
                native_rows.append({"fold": fold, "rmse_usd_per_oz": rmse, "mape_percent": mape})
                serialization_checks.append({"model": model, "fold": int(fold),
                    "max_csv_vs_float32_price_difference": float(np.max(np.abs(native_predictions - group.y_pred_price_original))),
                    "rmse_csv_minus_native": float(calculated.loc[calculated.fold == fold, "rmse_usd_per_oz"].iloc[0] - rmse),
                    "mape_csv_minus_native": float(calculated.loc[calculated.fold == fold, "mape_percent"].iloc[0] - mape)})
            logged_calculated = pd.DataFrame(native_rows)
        for key, label in (("rmse_usd_per_oz", "RMSE (USD/oz)"), ("mape_percent", "MAPE (%)")):
            np.testing.assert_allclose([logged_calculated[key].mean(), logged_calculated[key].std(ddof=0)],
                                       saved.loc[label, ["mean", "std"]].to_numpy(dtype=float),
                                       rtol=1e-12, atol=1e-12)
        log_path = table / f"{prefix}.log"
        evidence[str(log_path)] = sha(log_path)
        log_bytes = log_path.read_bytes()
        # Windows PowerShell redirected historical stdout as UTF-16 with a BOM.
        log_encoding = "utf-16" if log_bytes.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        log = log_bytes.decode(log_encoding)
        for row in logged_calculated.itertuples():
            expected = (f"Fold {row.fold} - RMSE (USD/oz): {row.rmse_usd_per_oz:.4f}, "
                        f"MAPE (%): {row.mape_percent:.4f}")
            assert expected in log, f"Log metric mismatch: {prefix} fold {row.fold}"

    fold_results = pd.DataFrame(rows)
    summary = fold_results.groupby("model", sort=False).agg(
        rmse_usd_per_oz_mean=("rmse_usd_per_oz", "mean"),
        rmse_usd_per_oz_std=("rmse_usd_per_oz", lambda x: np.std(x, ddof=0)),
        mape_percent_mean=("mape_percent", "mean"),
        mape_percent_std=("mape_percent", lambda x: np.std(x, ddof=0)),
    ).reset_index()
    retained_table = read_csv(table / "新Table8_回归尺度修复后.csv").set_index("Model")
    for row in summary.itertuples():
        assert retained_table.loc[row.model, "RMSE five-fold mean ± std (USD/oz)"] == (
            f"{row.rmse_usd_per_oz_mean:.4f} ± {row.rmse_usd_per_oz_std:.4f}")
        assert retained_table.loc[row.model, "MAPE five-fold mean ± std (%)"] == (
            f"{row.mape_percent_mean:.4f} ± {row.mape_percent_std:.4f}")

    # Explicitly preserve the distinction between five-fold means and pooled RMSE.
    fold_results.to_csv(output / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "five_fold_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pooled).to_csv(output / "pooled_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(scaler_checks).to_csv(output / "fold_scaler_verification.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(serialization_checks).to_csv(output / "csv_precision_verification.csv", index=False, encoding="utf-8-sig")
    # Hash the read-only evidence again to detect accidental modifications.
    assert all(sha(Path(path)) == value for path, value in evidence.items())
    report = {"passed": True, "mode": "recompute retained historical predictions; no training",
              "reference_root": str(reference), "n_samples": len(X), "n_features": X.shape[1],
              "n_predictions_each": len(test_indices), "five_folds": [len(t) for _, t in folds],
              "historical_label": "(GOLD.diff() > 0).astype(int)", "regression_label": "GOLD.shift(-1)",
              "classification_current_gold_reconstructable_from_features": True,
              "meta_training_labels_overlap_each_evaluation_fold": [len(np.intersect1d(test_indices, t)) for _, t in folds],
              "leakage_free_next_day_performance": False, "historical_files_unchanged": True,
              "scikit_learn_audit_version": sklearn.__version__,
              "random_forest_defaults_in_audit_environment": {
                  "classifier_max_features": RandomForestClassifier().max_features,
                  "regressor_max_features": RandomForestRegressor().max_features},
              "evidence_sha256": evidence}
    (output / "audit_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"PASS: 5 models, 25 folds, 10 saved target scalers; artifacts: {output}")


if __name__ == "__main__":
    main()
