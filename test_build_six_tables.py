"""The six-table gate must reject forecasts that have the right length but the wrong dates or prices."""

import unittest

import numpy as np
import pandas as pd

from build_six_tables import validate_targets
from protocol import EVAL_FOLDS, FOLD_SIZE


class TargetAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n = len(EVAL_FOLDS) * FOLD_SIZE
        cls.positions = np.arange(n)
        cls.expected = pd.DataFrame({
            "fold": np.repeat(EVAL_FOLDS, FOLD_SIZE),
            "date": pd.date_range("2000-01-01", periods=n).strftime("%Y-%m-%d"),
            "y_true_class": np.arange(n) % 2,
            "y_true_price": np.arange(n, dtype=float) + 100,
        })

    def frame(self):
        return self.expected.rename(columns={"y_true_price": "y_true_price_original"}).assign(
            row_index=self.positions)

    def validate(self, frame):
        validate_targets(frame, self.expected, self.positions,
                         price_column="y_true_price_original", class_column="y_true_class")

    def test_matching_daily_truth_passes(self):
        self.validate(self.frame())

    def test_same_length_wrong_daily_price_fails(self):
        frame = self.frame()
        frame.loc[100, "y_true_price_original"] += 1
        with self.assertRaisesRegex(ValueError, "gold prices"):
            self.validate(frame)

    def test_same_length_wrong_row_index_fails(self):
        frame = self.frame()
        frame.loc[100, "row_index"] += 1
        with self.assertRaisesRegex(ValueError, "row indices"):
            self.validate(frame)


if __name__ == "__main__":
    unittest.main()
