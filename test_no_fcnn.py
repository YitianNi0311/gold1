"""Checks on the local implementation with tiny fixtures; no search is run."""

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

import BFNE_Net_without_FCNNs as model
from regression_scale_utils import regression_metrics_original_price


ROOT = Path(__file__).resolve().parent


class NoFCNNTests(unittest.TestCase):
    def test_original_preprocessing_scaling_and_search_unchanged(self):
        original = ast.parse((ROOT / "BFNE_Net.py").read_text(encoding="utf-8"))
        local = ast.parse((ROOT / "BFNE_Net_without_FCNNs.py").read_text(encoding="utf-8"))
        original_functions = {node.name: node for node in original.body if isinstance(node, ast.FunctionDef)}
        local_functions = {node.name: node for node in local.body if isinstance(node, ast.FunctionDef)}
        for name in ("preprocess_data", "preprocess_and_scale", "preprocess_and_scale_reg_target",
                     "objective_lgbm_reg", "optimize_lgbm_reg", "objective_lgbm_clf", "optimize_lgbm_clf"):
            self.assertEqual(ast.dump(original_functions[name]), ast.dump(local_functions[name]), name)
        self.assertFalse(any(isinstance(node, ast.ClassDef) for node in local.body))
        imports = [node.module for node in ast.walk(local) if isinstance(node, ast.ImportFrom)]
        imports.extend(alias.name for node in ast.walk(local) if isinstance(node, ast.Import) for alias in node.names)
        self.assertNotIn("torch", imports)
        self.assertNotIn("BFNE_Net", imports)

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

if __name__ == "__main__":
    unittest.main()
