"""Local implementation checks; uses small fixtures without running the search."""

import ast
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import no_fcnn_model as model
import run_no_fcnn as runner
from regression_scale_utils import regression_metrics_original_price


ROOT = Path(__file__).resolve().parent


class NoFCNNTests(unittest.TestCase):
    def test_original_preprocessing_scaling_and_search_unchanged(self):
        original = ast.parse((ROOT / "归一.py").read_text(encoding="utf-8"))
        local = ast.parse((ROOT / "no_fcnn_model.py").read_text(encoding="utf-8"))
        original_functions = {node.name: node for node in original.body if isinstance(node, ast.FunctionDef)}
        local_functions = {node.name: node for node in local.body if isinstance(node, ast.FunctionDef)}
        for name in ("preprocess_data", "preprocess_and_scale", "preprocess_and_scale_reg_target",
                     "objective_lgbm_reg", "optimize_lgbm_reg", "objective_lgbm_clf", "optimize_lgbm_clf"):
            self.assertEqual(ast.dump(original_functions[name]), ast.dump(local_functions[name]), name)
        self.assertFalse(any(isinstance(node, ast.ClassDef) for node in local.body))
        imports = [node.module for node in ast.walk(local) if isinstance(node, ast.ImportFrom)]
        imports.extend(alias.name for node in ast.walk(local) if isinstance(node, ast.Import) for alias in node.names)
        self.assertNotIn("torch", imports)
        self.assertNotIn("归一", imports)

    def test_three_trees_and_original_parameters(self):
        best = dict(n_estimators=100, learning_rate=.05, num_leaves=20, max_depth=3,
                    subsample=.8, colsample_bytree=.8)
        with patch.object(model, "optimize_lgbm_reg", return_value=best), patch.object(model, "optimize_lgbm_clf", return_value=best):
            regressors = model.build_regressors(np.ones((8, 2)), np.arange(8.))
            classifiers = model.build_classifiers(np.ones((8, 2)), np.array([0, 1] * 4))
        for collection, names in (
            (regressors, ["LGBMRegressor", "XGBRegressor", "RandomForestRegressor"]),
            (classifiers, ["LGBMClassifier", "XGBClassifier", "RandomForestClassifier"]),
        ):
            self.assertEqual([type(item).__name__ for item in collection], names)
            xgb, forest = collection[1].get_params(), collection[2].get_params()
            for key, expected in dict(n_estimators=300, learning_rate=.03, max_depth=7,
                                      subsample=.9, colsample_bytree=.9, random_state=42).items():
                self.assertEqual(xgb[key], expected)
            self.assertEqual((forest["n_estimators"], forest["max_depth"], forest["random_state"]), (200, 10, 42))
        self.assertEqual(regressors[2].max_features, 1.0)
        self.assertEqual(classifiers[2].max_features, "sqrt")

    def test_folds_fit_only_training_prices_and_have_three_six_meta_columns(self):
        X = pd.DataFrame({"value": np.arange(12.)})
        classes, prices = pd.Series([0, 1] * 6), pd.Series(np.arange(12.) * 10 + 100)
        fd = model.prepare_fold(X, classes, prices, np.arange(8), np.arange(8, 12))
        self.assertAlmostEqual(fd["scaler_y"].mean_[0], prices.iloc[:8].mean())
        regressors = [Mock() for _ in range(3)]
        classifiers = [Mock() for _ in range(3)]
        for index, item in enumerate(regressors):
            item.predict.return_value = np.full(4, index)
        for item in classifiers:
            item.predict_proba.return_value = np.tile([.3, .7], (4, 1))
        with patch.object(model, "build_regressors", return_value=regressors), patch.object(model, "build_classifiers", return_value=classifiers):
            reg_features, clf_features = model.train_fold(fd)
        self.assertEqual(reg_features.shape, (4, 3))
        self.assertEqual(clf_features.shape, (4, 6))
        for item in regressors:
            np.testing.assert_array_equal(item.fit.call_args.args[1], fd["yr_train_fit"])
        for item in classifiers:
            np.testing.assert_array_equal(item.fit.call_args.args[1], classes.iloc[:8])

    def test_local_pipeline_outputs_and_single_fold_inverse(self):
        X = pd.DataFrame({"value": np.arange(60.)})
        classes, prices = pd.Series([0, 1] * 30), pd.Series(np.arange(60.) * 10 + 100)
        dates = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=60)})

        def fixture_features(fd):
            reg = np.repeat(fd["Xr_test"][:, :1], 3, axis=1)
            prob = .2 + .6 * fd["yc_test"].to_numpy()
            return reg, np.tile(np.column_stack([1 - prob, prob]), (1, 3))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch.object(runner, "preprocess_data", return_value=(X, classes, prices)), \
                 patch.object(runner.pd, "read_excel", return_value=dates), \
                 patch.object(runner, "train_fold", side_effect=fixture_features) as training, \
                 patch.object(runner, "regression_metrics_original_price", wraps=regression_metrics_original_price) as metrics, \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.run(Path(directory) / "fixture.xlsx", output)
            self.assertEqual(training.call_count, 10)
            self.assertEqual(metrics.call_count, 6)
            self.assertEqual(sum("prediction_scaler" in call.kwargs for call in metrics.call_args_list), 5)
            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["fcnn_count"], 0)
            self.assertEqual(manifest["n_predictions"], 50)
            predictions = pd.read_csv(output / "predictions.csv")
            np.testing.assert_allclose(predictions.y_true_price_original, prices.iloc[10:])
            self.assertEqual(predictions.groupby("fold").size().tolist(), [10] * 5)
            fold_metrics = pd.read_csv(output / "fold_metrics.csv").set_index("fold")
            for fold, frame in predictions.groupby("fold"):
                _, rmse, mape = regression_metrics_original_price(frame.y_true_price_original, frame.y_pred_price_original)
                self.assertAlmostEqual(fold_metrics.loc[fold, "rmse_usd_per_oz"], rmse)
                self.assertAlmostEqual(fold_metrics.loc[fold, "mape_percent"], mape)
            summary = pd.read_csv(output / "metric_summary.csv").set_index("metric")
            self.assertEqual(summary.loc["mape_percent", "label"], "MAPE (%)")
            self.assertEqual(summary.loc["rmse_usd_per_oz", "label"], "RMSE (USD/oz)")
            self.assertEqual(len(list(output.glob("scaler_y_fold*.json"))), 5)


if __name__ == "__main__":
    unittest.main()
