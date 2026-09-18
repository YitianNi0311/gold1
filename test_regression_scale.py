import ast
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from regression_scale_utils import regression_metrics_original_price


ROOT = Path(__file__).resolve().parent


def source_function(name, function, namespace):
    tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    exec(compile(ast.Module(body=[node], type_ignores=[]), name, "exec"), namespace)
    return namespace[function]


class RegressionScaleTests(unittest.TestCase):
    def test_mape_is_percentage_once(self):
        true = np.array([100.0, 200.0])
        pred = np.array([110.0, 180.0])
        restored, rmse, mape = regression_metrics_original_price(true, pred)
        np.testing.assert_allclose(restored, pred)
        self.assertAlmostEqual(rmse, np.sqrt(250.0))
        self.assertAlmostEqual(mape, 10.0)

    def test_two_folds_use_their_own_scalers(self):
        folds = [
            (np.array([80.0, 100.0, 120.0]), np.array([90.0, 110.0]), np.array([95.0, 100.0])),
            (np.array([900.0, 1200.0, 1500.0]), np.array([1050.0, 1350.0]), np.array([1100.0, 1300.0])),
        ]
        for train, true, pred_price in folds:
            scaler = StandardScaler().fit(train.reshape(-1, 1))
            pred_scaled = scaler.transform(pred_price.reshape(-1, 1)).reshape(-1)
            restored, rmse_scaled_path, mape_scaled_path = regression_metrics_original_price(
                true, pred_scaled, prediction_scaler=scaler
            )
            _, rmse_direct, mape_direct = regression_metrics_original_price(true, pred_price)
            np.testing.assert_allclose(restored, pred_price)
            self.assertAlmostEqual(rmse_scaled_path, rmse_direct)
            self.assertAlmostEqual(mape_scaled_path, mape_direct)

    def test_static_final_paths(self):
        bfne = (ROOT / "归一.py").read_text(encoding="utf-8")
        self.assertEqual(bfne.count("regression_metrics_original_price("), 5)
        self.assertNotIn(
            "mean_absolute_percentage_error_custom(y_test_reg_scaled, fold_meta_preds_reg_scaled)",
            bfne,
        )
        for name in ("RF.py", "XGBoost.py", "gold lightgbm.py"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("regression_metrics_original_price(", text)
            self.assertIn("RMSE (USD/oz)", text)
            self.assertIn("MAPE (%)", text)

    def test_local_runner_scale_and_zero_fcnn(self):
        runner = (ROOT / "run_no_fcnn.py").read_text(encoding="utf-8")
        self.assertIn(
            'fd["yr_test"], pred_fit, prediction_scaler=fd["scaler_y"]',
            runner,
        )
        self.assertNotIn(
            'fd["scaler_y"].inverse_transform(pred_fit.reshape(-1, 1))',
            runner,
        )
        self.assertIn('mape_percent=mape', runner)
        self.assertIn("from no_fcnn_model import", runner)
        zero = ast.parse((ROOT / "no_fcnn_model.py").read_text(encoding="utf-8"))
        for name, expected in (
            ("build_regressors", ["LGBMRegressor", "XGBRegressor", "RandomForestRegressor"]),
            ("build_classifiers", ["LGBMClassifier", "XGBClassifier", "RandomForestClassifier"]),
        ):
            function = next(n for n in zero.body if isinstance(n, ast.FunctionDef) and n.name == name)
            returned = next(n for n in function.body if isinstance(n, ast.Return))
            self.assertEqual([n.func.id for n in returned.value.elts], expected)

    def test_every_final_block_restores_once_with_its_fold_scaler(self):
        tree = ast.parse((ROOT / "归一.py").read_text(encoding="utf-8"))
        assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                       and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                       and n.value.func.id == "regression_metrics_original_price"]
        self.assertEqual(len(assignments), 5)
        for block, assignment in enumerate(assignments, 1):
            for train in ([80., 100., 120.], [900., 1200., 1500.]):
                with self.subTest(block=block, train=train):
                    scaler = StandardScaler().fit(np.array(train).reshape(-1, 1))
                    scaled = scaler.transform(np.array([110., 180.]).reshape(-1, 1)).ravel()
                    spy = Mock(wraps=scaler)
                    namespace = dict(regression_metrics_original_price=regression_metrics_original_price,
                                     y_test_reg_fold=pd.Series([100., 200.]),
                                     fold_meta_preds_reg_scaled=scaled, scaler_y_reg_eval=spy)
                    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "final_block", "exec"), namespace)
                    np.testing.assert_allclose(namespace["fold_meta_preds_reg"], [110., 180.])
                    self.assertAlmostEqual(namespace["rmse"], np.sqrt(250.))
                    self.assertAlmostEqual(namespace["mape"], 10.)
                    spy.inverse_transform.assert_called_once()

    def test_tree_raw_targets_and_console_csv_units(self):
        for filename, function, prefix in (
            ("RF.py", "train_random_forest", "RF"),
            ("XGBoost.py", "train_xgboost", "XGBoost"),
            ("gold lightgbm.py", "train_lightgbm", "LightGBM"),
        ):
            with self.subTest(filename=filename):
                fitted = []

                class Classifier:
                    def __init__(self, **kwargs):
                        pass

                    def fit(self, x, y):
                        return self

                    def predict(self, x):
                        return np.array([0, 1])

                    def predict_proba(self, x):
                        return np.array([[.8, .2], [.2, .8]])

                class Regressor(Classifier):
                    def fit(self, x, y):
                        fitted.append(np.asarray(y).copy())
                        return self

                    def predict(self, x):
                        return np.array([110., 180.])

                namespace = {name: getattr(metrics, name) for name in
                             ("accuracy_score", "precision_score", "recall_score", "f1_score", "roc_auc_score")}
                helper = Mock(wraps=regression_metrics_original_price)
                namespace.update(np=np, pd=pd, Path=Path, __file__=str(ROOT / filename),
                                 regression_metrics_original_price=helper,
                                 RandomForestClassifier=Classifier, RandomForestRegressor=Regressor,
                                 XGBClassifier=Classifier, XGBRegressor=Regressor,
                                 lgb=type("LightGBM", (), {"LGBMClassifier": Classifier, "LGBMRegressor": Regressor}))
                train = source_function(filename, function, namespace)
                raw_train = pd.Series([80., 120.])
                result = train(np.zeros((2, 1)), pd.Series([0, 1]), raw_train,
                               np.zeros((2, 1)), pd.Series([0, 1]), pd.Series([100., 200.]))
                np.testing.assert_array_equal(fitted[0], raw_train)
                self.assertEqual(helper.call_args.kwargs, {})
                self.assertAlmostEqual(result[5], np.sqrt(250.))
                self.assertAlmostEqual(result[6], 10.)
                from sklearn.model_selection import TimeSeriesSplit
                namespace.update(TimeSeriesSplit=TimeSeriesSplit, StandardScaler=StandardScaler,
                                 preprocess_data=lambda path: (pd.DataFrame({"x": range(12)}),
                                                               pd.Series([0, 1] * 6), pd.Series([100., 200.] * 6)))
                namespace[function] = lambda *args: result
                main = source_function(filename, "main", namespace)
                previous = Path.cwd()
                with tempfile.TemporaryDirectory() as directory:
                    output = io.StringIO()
                    try:
                        os.chdir(directory)
                        with contextlib.redirect_stdout(output):
                            main()
                        summary = pd.read_csv(f"{prefix}_metric_summary.csv").set_index("metric")
                        predictions = pd.read_csv(f"{prefix}_regression_predictions_original_price.csv")
                    finally:
                        os.chdir(previous)
                self.assertIn("RMSE (USD/oz)", output.getvalue())
                self.assertIn("MAPE (%)", output.getvalue())
                self.assertAlmostEqual(summary.loc["MAPE (%)", "mean"], 10.)
                self.assertAlmostEqual(summary.loc["RMSE (USD/oz)", "mean"], np.sqrt(250.))
                self.assertEqual(len(predictions), 10)
                np.testing.assert_array_equal(predictions.y_true_price_original, [100., 200.] * 5)
                np.testing.assert_array_equal(predictions.y_pred_price_original, [110., 180.] * 5)


if __name__ == "__main__":
    unittest.main()
