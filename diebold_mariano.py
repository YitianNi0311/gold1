"""Paired Diebold-Mariano comparisons for full BFNE versus zero FCNN.

The test uses one forecast per historical date and fold. A positive loss
difference means that the full model has the larger loss. Historical label
and meta-evaluation leakage is retained; the resulting p-values must not
be interpreted as evidence of leakage-free forecasting skill.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


KEYS = ("fold", "date", "row_index")
TARGETS = ("y_true_class", "y_true_price_original")
PREDICTIONS = ("y_pred_class", "y_pred_probability", "y_pred_price_original")
LOSSES = ("squared_price_error", "absolute_price_error",
          "absolute_percentage_error_percent", "brier", "zero_one")


def dm_test(loss_full, loss_zero, *, lag=10):
    """Two-sided modified DM statistic with Bartlett HAC and h=1 correction.

    Losses are paired in chronological order. The pre-specified lag is in
    trading observations; it is not selected from the observed p-value.
    """
    full = np.asarray(loss_full, dtype=float).reshape(-1)
    zero = np.asarray(loss_zero, dtype=float).reshape(-1)
    if full.shape != zero.shape or not np.isfinite(full).all() or not np.isfinite(zero).all():
        raise ValueError("Paired loss series must have equal length and finite values")
    n = len(full)
    if not isinstance(lag, int) or lag < 0 or n <= lag + 1:
        raise ValueError("HAC lag must be nonnegative and shorter than the series")
    difference = full - zero
    centered = difference - difference.mean()
    long_run_variance = np.dot(centered, centered) / n
    for offset in range(1, lag + 1):
        covariance = np.dot(centered[offset:], centered[:-offset]) / n
        long_run_variance += 2 * (1 - offset / (lag + 1)) * covariance
    if not np.isfinite(long_run_variance) or long_run_variance <= 0:
        raise ValueError("Loss differential has no positive estimated long-run variance")
    # Harvey-Leybourne-Newbold small-sample correction for horizon h=1.
    correction = np.sqrt((n - 1) / n)
    statistic = float(correction * difference.mean() /
                      np.sqrt(long_run_variance / n))
    return {
        "n_dates": n,
        "mean_loss_full_minus_zero": float(difference.mean()),
        "dm_statistic": statistic,
        "p_value_two_sided": float(2 * student_t.sf(abs(statistic), df=n - 1)),
        "hac_lag": lag,
        "forecast_horizon": 1,
    }


def _read_predictions(path):
    frame = pd.read_csv(path, float_precision="round_trip")
    required = set(KEYS + TARGETS + PREDICTIONS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns in {path}: {sorted(missing)}")
    frame = frame.loc[:, list(KEYS + TARGETS + PREDICTIONS)].copy()
    frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    frame = frame.sort_values(list(KEYS)).reset_index(drop=True)
    if len(frame) != 3680 or frame.groupby("fold").size().to_dict() != dict.fromkeys(range(1, 6), 736):
        raise ValueError(f"Expected five folds of 736 predictions in {path}")
    if frame.duplicated(list(KEYS)).any() or frame.isna().any().any():
        raise ValueError(f"Duplicate or missing prediction values in {path}")
    numbers = frame.select_dtypes(include="number")
    if not np.isfinite(numbers.to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite prediction values in {path}")
    if (not frame.y_pred_probability.between(0, 1).all() or
            not frame.y_true_class.isin((0, 1)).all() or
            not frame.y_pred_class.isin((0, 1)).all()):
        raise ValueError(f"Invalid class labels or probabilities in {path}")
    if not frame.y_true_price_original.gt(0).all():
        raise ValueError(f"Nonpositive true gold price in {path}")
    return frame


def paired_losses(full_path, zero_path, reference_path):
    """Reject any fold/date/row/target mismatch before computing losses."""
    full = _read_predictions(full_path)
    zero = _read_predictions(zero_path)
    if not full.loc[:, KEYS].equals(zero.loc[:, KEYS]):
        raise ValueError("Full and zero-FCNN test positions differ")
    if not full.y_true_class.equals(zero.y_true_class) or not np.allclose(
            full.y_true_price_original, zero.y_true_price_original, rtol=0, atol=1e-8):
        raise ValueError("Full and zero-FCNN true targets differ")
    reference = pd.read_csv(reference_path, float_precision="round_trip")
    reference["date"] = pd.to_datetime(reference.date).dt.strftime("%Y-%m-%d")
    reference = reference.sort_values(["fold", "date"]).reset_index(drop=True)
    if len(reference) != len(full) or not reference[["fold", "date"]].equals(
            full[["fold", "date"]]) or not reference.y_true_class.equals(
            full.y_true_class) or not np.allclose(
            reference.y_true_price, full.y_true_price_original, rtol=0, atol=1e-8):
        raise ValueError("Predictions do not match the saved historical test targets")
    output = full.loc[:, KEYS].copy()
    truth = full.y_true_price_original.to_numpy(dtype=float)
    labels = full.y_true_class.to_numpy(dtype=int)
    for suffix, frame in (("full", full), ("zero", zero)):
        forecast = frame.y_pred_price_original.to_numpy(dtype=float)
        error = truth - forecast
        output[f"squared_price_error_{suffix}"] = error ** 2
        output[f"absolute_price_error_{suffix}"] = np.abs(error)
        output[f"absolute_percentage_error_percent_{suffix}"] = np.abs(error / truth) * 100
        output[f"brier_{suffix}"] = (labels - frame.y_pred_probability.to_numpy(dtype=float)) ** 2
        output[f"zero_one_{suffix}"] = (labels != frame.y_pred_class.to_numpy(dtype=int)).astype(float)
    return output


def summarize(loss_frame, *, lag=10):
    rows = []
    for loss in LOSSES:
        row = {"loss": loss, **dm_test(loss_frame[f"{loss}_full"],
                                       loss_frame[f"{loss}_zero"], lag=lag)}
        rows.append(row)
    return pd.DataFrame(rows)


def average_seed_losses(seed_frames):
    """Average paired losses by date before testing; never average p-values."""
    if set(seed_frames) != {42, 43, 44}:
        raise ValueError("The aggregate requires complete seeds 42, 43 and 44")
    reference = seed_frames[42].loc[:, KEYS]
    for seed in (43, 44):
        if not seed_frames[seed].loc[:, KEYS].equals(reference):
            raise ValueError(f"Seed {seed} has different test positions")
    result = reference.copy()
    for loss in LOSSES:
        for model in ("full", "zero"):
            column = f"{loss}_{model}"
            result[column] = np.mean(
                [seed_frames[seed][column].to_numpy(dtype=float)
                 for seed in (42, 43, 44)], axis=0)
    return result


def save_dm_tables(seed_prediction_pairs, reference_path, output_dir, *, lag=10):
    """Save seed-42 and three-seed DM tables plus auditable daily losses."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    frames = {seed: paired_losses(full, zero, reference_path)
              for seed, (full, zero) in seed_prediction_pairs.items()}
    aggregate = average_seed_losses(frames)
    tables = {"seed42": summarize(frames[42], lag=lag),
              "three_seed": summarize(aggregate, lag=lag)}
    output_dir.mkdir(parents=True)
    for seed, frame in frames.items():
        frame.to_csv(output_dir / f"daily_loss_seed{seed}.csv", index=False,
                     encoding="utf-8-sig", float_format="%.17g")
    aggregate.to_csv(output_dir / "daily_loss_three_seed_average.csv", index=False,
                     encoding="utf-8-sig", float_format="%.17g")
    for label, table in tables.items():
        table.to_csv(output_dir / f"dm_{label}.csv", index=False,
                     encoding="utf-8-sig", float_format="%.17g")
    return tables
