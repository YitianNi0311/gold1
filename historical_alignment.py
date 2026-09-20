"""Check that the rebuilt baselines line up with the seed-42 historical test rows (read-only).

The old labels are kept as they were. Passing only means the test rows match; it does
not remove the label or stacking leakage."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from BFNE_Net_without_FCNNs import preprocess_data


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "GOLD_cleaned.xlsx"
REFERENCE = ROOT / "seed42_reference_targets.csv"
REFERENCE_MANIFEST = ROOT / "seed42_reference_targets_manifest.json"


def aligned_data(data_path=DATA, reference_path=REFERENCE):
    """Return the 4,419 original rows, but only after every test target checks out."""
    data_path, reference_path = Path(data_path), Path(reference_path)
    x, y_class, y_price = preprocess_data(data_path)
    if len(x) != 4419 or x.shape[1] != 48:
        raise ValueError(f"Expected 4,419 rows and 48 features, got {x.shape}")
    raw = pd.read_excel(data_path)
    dates = pd.to_datetime(raw.loc[x.index, "Date"]).reset_index(drop=True)
    expected = []
    splits = list(TimeSeriesSplit(n_splits=5).split(x))
    for fold, (_, test) in enumerate(splits, 1):
        if len(test) != 736:
            raise ValueError(f"Fold {fold} has {len(test)} test rows, expected 736")
        expected.append(pd.DataFrame({
            "fold": fold,
            "date": dates.iloc[test].dt.strftime("%Y-%m-%d").to_numpy(),
            "y_true_class": np.asarray(y_class.iloc[test], dtype=int),
            "y_true_price": np.asarray(y_price.iloc[test], dtype=float),
        }))
    expected = pd.concat(expected, ignore_index=True)
    if reference_path.resolve() == REFERENCE.resolve():
        manifest = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
        digest = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        if digest != manifest["reference_sha256"]:
            raise ValueError("Bundled historical target reference checksum differs")
    saved = pd.read_csv(reference_path, float_precision="round_trip")
    columns = ["fold", "date", "y_true_class", "y_true_price"]
    if len(expected) != 3680 or len(saved) != 3680:
        raise ValueError("Expected exactly 3,680 historical test predictions")
    if saved[columns[:3]].astype(str).reset_index(drop=True).equals(
            expected[columns[:3]].astype(str)) is False:
        raise ValueError("Historical fold, date, or classification labels differ")
    if not np.allclose(saved.y_true_price, expected.y_true_price, rtol=0, atol=1e-8):
        raise ValueError("Historical true gold prices differ")
    if expected.duplicated(["fold", "date"]).any():
        raise ValueError("Duplicate test dates")
    return x, y_class, y_price, dates, splits, expected
