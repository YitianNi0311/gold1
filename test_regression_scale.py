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
        runner = (ROOT / "run_seeded_ensembles.py").read_text(encoding="utf-8")
        self.assertIn('fd["yr_test"], meta_reg.predict(block["reg"])', runner)
        for name in ("Random_Forest.py", "XGBoost.py", "LightGBM.py"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("regression_metrics_original_price(", text)
            self.assertIn("RMSE (USD/oz)", text)
            self.assertIn("MAPE (%)", text)

    def test_tree_raw_targets_and_console_csv_units(self):
        for filename, function, prefix in (
            ("Random_Forest.py", "train_random_forest", "RF"),
            ("XGBoost.py", "train_xgboost", "XGBoost"),
            ("LightGBM.py", "train_lightgbm", "LightGBM"),
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
