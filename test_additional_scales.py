"""Synthetic checks of the evaluation scale in a few legacy regression scripts."""

import ast
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

from regression_scale_utils import regression_metrics_original_price


ROOT = Path(__file__).resolve().parent


class AdditionalScaleTests(unittest.TestCase):
    def test_scaled_one_fcnn_variants_restore_each_fold_once(self):
        for name in ("去2FCNN.py", "去掉2个.py"):
            tree = ast.parse((ROOT / "archive" / name).read_text(encoding="utf-8"))
            matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                       and isinstance(node.value, ast.Call)
                       and isinstance(node.value.func, ast.Name)
                       and node.value.func.id == "regression_metrics_original_price"]
            self.assertEqual(len(matches), 1, name)
            statement = ast.fix_missing_locations(ast.Module(body=matches, type_ignores=[]))
            for training_prices in ([0, 100], [1000, 2000]):
                scaler = StandardScaler().fit(np.asarray(training_prices).reshape(-1, 1))
                namespace = {
                    "regression_metrics_original_price": regression_metrics_original_price,
                    "y_test_reg_fold": np.array([100., 200.]),
                    "fold_meta_preds_reg": scaler.transform(
                        np.array([110., 180.]).reshape(-1, 1)).ravel(),
                    "scaler_y_reg": scaler,
                }
                exec(compile(statement, name, "exec"), namespace)
                np.testing.assert_allclose(namespace["fold_meta_preds_reg"], [110, 180])
                self.assertAlmostEqual(namespace["mape"], 10.0)
                self.assertAlmostEqual(namespace["rmse"], np.sqrt(250.0))

    def test_native_price_variants_return_percent(self):
        for name in ("去no1fcnn.py", "加上软投票.py", "最后的尝试.py",
                     "feature_analysis.py", "不拆分.py"):
            tree = ast.parse((ROOT / "archive" / name).read_text(encoding="utf-8"))
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                            and node.name == "mean_absolute_percentage_error_custom")
            namespace = {"mean_absolute_percentage_error": mean_absolute_percentage_error}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                         name, "exec"), namespace)
            self.assertAlmostEqual(namespace[function.name]([100, 200], [110, 180]), 10.0,
                                   msg=name)


if __name__ == "__main__":
    unittest.main()
