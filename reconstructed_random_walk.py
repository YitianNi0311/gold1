"""New persistence baseline, not a recovered author implementation.

Predict GOLD_(t+1) with GOLD_t on the historical 4,419-row test schedule.
This script writes a separate run directory and never changes Table 8/9.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from historical_alignment import DATA, ROOT, REFERENCE, aligned_data
from regression_scale_utils import regression_metrics_original_price


def run(output_dir):
    output_dir = Path(output_dir)
    x, _, y_price, dates, splits, expected = aligned_data()
    raw = pd.read_excel(DATA)
    gold_today = raw.loc[x.index, "GOLD"].to_numpy(dtype=float)
    frames, metrics = [], []
    for fold, (_, test) in enumerate(splits, 1):
        predictions = gold_today[test]
        truth = y_price.iloc[test].to_numpy(dtype=float)
        _, rmse, mape = regression_metrics_original_price(truth, predictions)
        frame = expected.loc[expected.fold.eq(fold)].copy()
        frame["y_pred_price_original"] = predictions
        frames.append(frame)
        metrics.append({"fold": fold, "rmse_usd_per_oz": rmse, "mape_percent": mape})
    all_predictions = pd.concat(frames, ignore_index=True)
    if not np.array_equal(all_predictions.date.to_numpy(), expected.date.to_numpy()):
        raise ValueError("Random Walk test dates do not match the historical run")
    output_dir.mkdir(parents=True, exist_ok=False)
    all_predictions.to_csv(output_dir / "predictions.csv", index=False,
                           encoding="utf-8-sig", float_format="%.17g")
    pd.DataFrame(metrics).to_csv(output_dir / "fold_metrics.csv", index=False,
                                 encoding="utf-8-sig", float_format="%.17g")
    (output_dir / "run_manifest.json").write_text(json.dumps({
        "status": "completed", "source": "newly reconstructed persistence baseline",
        "author_implementation_recovered": False, "seed": 42,
        "formula": "prediction at row t = GOLD_t; target = GOLD_(t+1)",
        "n_cleaned_rows": len(x), "n_test_predictions": len(all_predictions),
        "reference": str(REFERENCE),
        "historical_classification_label": "(GOLD.diff() > 0)",
        "historical_leakage_note": "The existing Table 8/9 results retain known label and meta-evaluation leakage.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reconstructed_runs" /
                        f"random_walk_{datetime.now():%Y%m%d_%H%M%S}")
    args = parser.parse_args()
    print(run(args.output))
