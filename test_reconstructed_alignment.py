"""Preflight on the real data for the rebuilt baselines. No training."""

import unittest

from historical_alignment import aligned_data
from BFNE_Net_without_FE import RAW_FEATURES, audit_features


class ReconstructedAlignmentTests(unittest.TestCase):
    def test_historical_rows_and_table1_features(self):
        x, y_class, y_price, dates, splits, expected = aligned_data()
        self.assertEqual((len(x), x.shape[1], len(expected)), (4419, 48, 3680))
        self.assertEqual([len(test) for _, test in splits], [736] * 5)
        selected, deleted = audit_features(x)
        self.assertEqual(tuple(selected.columns), RAW_FEATURES)
        self.assertEqual((selected.shape[1], len(deleted)), (11, 37))
        self.assertEqual(len(y_class), len(y_price))
        self.assertEqual(len(dates), len(x))


if __name__ == "__main__":
    unittest.main()
