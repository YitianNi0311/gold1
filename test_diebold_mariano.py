"""Numeric and alignment tests for the paired forecast comparison."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from protocol import EVAL_FOLDS, FOLD_SIZE
from diebold_mariano import average_seed_losses, dm_test, paired_losses, summarize


class DieboldMarianoTests(unittest.TestCase):
    def test_dm_sign_and_p_value_for_correlated_loss_series(self):
        rng = np.random.default_rng(7)
        innovation = rng.normal(size=600)
        correlated = np.empty_like(innovation)
        correlated[0] = innovation[0]
        for index in range(1, len(innovation)):
            correlated[index] = 0.5 * correlated[index - 1] + innovation[index]
        full = 3 + correlated
        zero = 2 + 0.2 * correlated
        result = dm_test(full, zero, lag=10)
        self.assertEqual(result["n_dates"], 600)
        self.assertGreater(result["mean_loss_full_minus_zero"], 0)
        self.assertGreater(result["dm_statistic"], 0)
        self.assertGreaterEqual(result["p_value_two_sided"], 0)
        self.assertLessEqual(result["p_value_two_sided"], 1)
        with self.assertRaises(ValueError):
            dm_test(np.ones(50), np.ones(50))

    def test_prediction_gate_checks_dates_targets_and_three_seeds(self):
        n = len(EVAL_FOLDS) * FOLD_SIZE
        dates = pd.date_range("2006-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
        gold = 100 + np.arange(n) * 0.1
        labels = np.arange(n) % 2
        base = pd.DataFrame({
            "fold": np.repeat(EVAL_FOLDS, FOLD_SIZE), "date": dates,
            "row_index": np.arange(n) + 30,
            "y_true_class": labels, "y_true_price_original": gold,
            "y_pred_class": labels,
            "y_pred_probability": np.where(labels == 1, 0.8, 0.2),
            "y_pred_price_original": gold + 5 + np.sin(np.arange(n) / 11),
        })
        zero = base.copy()
        zero["y_pred_price_original"] = gold + 2 + np.sin(np.arange(n) / 13)
        zero["y_pred_probability"] = (np.where(labels == 1, 0.9, 0.1)
                                      + 0.04 * np.sin(np.arange(n) / 7))
        zero["y_pred_class"] = np.where(np.arange(n) % 19 == 0, 1 - labels, labels)
        reference = base[["fold", "date", "y_true_class"]].copy()
        reference["y_true_price"] = gold
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_path, zero_path, ref_path = [root / f for f in
                                              ("full.csv", "zero.csv", "reference.csv")]
            base.to_csv(full_path, index=False)
            zero.to_csv(zero_path, index=False)
            reference.to_csv(ref_path, index=False)
            losses = paired_losses(full_path, zero_path, ref_path)
            self.assertEqual(len(losses), n)
            self.assertEqual(len(summarize(losses)), 5)
            mean_losses = average_seed_losses({42: losses, 43: losses, 44: losses})
            pd.testing.assert_frame_equal(mean_losses, losses[mean_losses.columns])
            zero.loc[0, "date"] = "2000-01-01"
            zero.to_csv(zero_path, index=False)
            with self.assertRaisesRegex(ValueError, "positions differ"):
                paired_losses(full_path, zero_path, ref_path)


if __name__ == "__main__":
    unittest.main()
