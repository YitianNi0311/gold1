"""The meta model that scores a fold may only be trained on earlier folds."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_seeded_ensembles as runner
from historical_alignment import aligned_data


def blocks():
    return [dict(fold=k, reg=np.full((4, 3), float(k)), clf=np.full((4, 6), float(k)),
                 reg_labels=np.full(4, float(k)), clf_labels=np.full(4, k))
            for k in range(1, 6)]


class ForwardProtocolTests(unittest.TestCase):
    def test_meta_models_only_see_earlier_blocks(self):
        for fold in range(2, 6):
            with patch.object(runner, "meta_models", return_value=("reg", "clf")) as meta:
                runner.forward_meta_models(blocks(), fold, seed=42)
            reg_features, reg_labels, clf_features, clf_labels, _ = meta.call_args.args
            self.assertEqual(len(reg_features), fold - 1)
            self.assertLess(max(np.max(f) for f in reg_features), fold)
            self.assertLess(reg_labels.max(), fold)
            self.assertLess(clf_labels.max(), fold)

    def test_first_block_has_nothing_to_train_on(self):
        with self.assertRaises(ValueError):
            runner.forward_meta_models(blocks(), 1, seed=42)

    def test_label_is_next_day_direction_and_not_derivable_from_today(self):
        x, y_class, y_price, _, splits, expected = aligned_data()
        gold_today = pd.read_excel(runner.DATA).loc[x.index, "GOLD"].to_numpy(dtype=float)
        np.testing.assert_array_equal(y_class.to_numpy(), (y_price.to_numpy() > gold_today).astype(int))
        # the old label (today vs yesterday) equals lag1 < today, which the features can reconstruct
        old = (gold_today > x["GOLD_lag1"].to_numpy()).astype(int)
        self.assertLess(abs(float(np.mean(old == y_class.to_numpy())) - 0.5), 0.05)
        first_test = splits[0][1]
        self.assertGreater(splits[1][1].min(), first_test.max())


if __name__ == "__main__":
    unittest.main()
