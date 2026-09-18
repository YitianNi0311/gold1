"""Build six new tables only after all 42/43/44 aligned runs are complete."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diebold_mariano import save_dm_tables
from historical_alignment import REFERENCE, aligned_data


MODELS = (
    ("random_forest", "Random Forest", "original project tree script"),
    ("xgboost", "XGBoost", "original project tree script"),
    ("lightgbm", "LightGBM", "original project tree script"),
    ("random_walk", "Random Walk", "new persistence reconstruction"),
    ("cnn_lstm", "CNN-LSTM", "aligned classifier and new regression head"),
    ("without_fe", "BFNE-Net without FE", "new paper-defined reconstruction"),
    ("no_fcnn", "BFNE-Net without FCNNs", "three-tree implementation"),
    ("full", "BFNE-Net", "full implementation"),
)


def validate_targets(frame, expected, expected_positions, *, price_column, class_column):
    if len(frame) != 3680 or not np.array_equal(frame.fold.to_numpy(), expected.fold.to_numpy()) or not np.array_equal(
            pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d"), expected.date):
        raise ValueError("Fold or daily test dates differ from the historical reference")
    if not np.array_equal(frame[class_column].to_numpy(dtype=int), expected.y_true_class.to_numpy(dtype=int)):
        raise ValueError("Daily classification truths differ from the historical reference")
    if not np.allclose(frame[price_column], expected.y_true_price, rtol=0, atol=1e-8):
        raise ValueError("Daily original gold prices differ from the historical reference")
    if "row_index" in frame and not np.array_equal(
            frame.row_index.to_numpy(dtype=int), expected_positions):
        raise ValueError("Daily source row indices differ from the historical reference")


def read_metrics(run_root, model, seed, expected, expected_positions):
    folder = run_root / f"seed{seed}" / model
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or manifest.get("seed") != seed:
        raise ValueError(f"Incomplete run: {folder}")
    if model == "cnn_lstm":
        reg = pd.read_csv(folder / "regression_fold_metrics.csv")
        clf = pd.read_csv(folder / "classification_fold_metrics.csv")
        metrics = reg.merge(clf[["fold", "accuracy", "precision", "recall", "f1", "auc"]],
                            on="fold", validate="one_to_one")
        regression = pd.read_csv(folder / "regression_predictions.csv")
        classification = pd.read_csv(folder / "classification_predictions.csv")
        validate_targets(regression, expected, expected_positions,
                         price_column="y_true_price", class_column="y_true_class")
        validate_targets(classification, expected, expected_positions,
                         price_column="y_true_price", class_column="y_true")
    else:
        metrics = pd.read_csv(folder / "fold_metrics.csv")
        if "round" in metrics:
            metrics = metrics.loc[metrics["round"].eq(1)]
        predictions = pd.read_csv(folder / "predictions.csv")
        validate_targets(predictions, expected, expected_positions,
                         price_column="y_true_price_original", class_column="y_true_class")
    if metrics.groupby("fold").size().to_dict() != dict.fromkeys(range(1, 6), 1):
        raise ValueError(f"Incorrect five-fold metrics: {folder}")
    if len(metrics) != 5 or metrics.fold.tolist() != [1, 2, 3, 4, 5]:
        raise ValueError(f"Expected five chronological folds: {folder}")
    return metrics


def format_mean_std(values):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.4f} ± {values.std(ddof=0):.4f}"


def make_tables(run_root, seeds, expected, expected_positions):
    collected = {}
    for model, _, _ in MODELS:
        for seed in seeds:
            collected[model, seed] = read_metrics(
                run_root, model, seed, expected, expected_positions)
    table8, table9 = [], []
    for model, label, source in MODELS:
        metrics = pd.concat([collected[model, seed] for seed in seeds], ignore_index=True)
        rmse_seed_means = [collected[model, seed].rmse_usd_per_oz.mean() for seed in seeds]
        mape_seed_means = [collected[model, seed].mape_percent.mean() for seed in seeds]
        if len(seeds) == 1:
            rmse_display = format_mean_std(metrics.rmse_usd_per_oz)
            mape_display = format_mean_std(metrics.mape_percent)
        else:
            rmse_display = format_mean_std(rmse_seed_means)
            mape_display = format_mean_std(mape_seed_means)
        table8.append({
            "Model": label,
            "Directional AUC mean": float(metrics.auc.mean()) if model != "random_walk" else np.nan,
            "RMSE mean ± std (USD/oz)": rmse_display,
            "MAPE mean ± std (%)": mape_display,
            "N predictions per seed": 3680,
            "N seeds": len(seeds),
            "Result source": source,
            "Status": "new aligned rerun; historical leakage retained",
        })
        if model != "random_walk":
            table9.append({
                "Model": label,
                "Accuracy": float(metrics.accuracy.mean()),
                "Precision": float(metrics.precision.mean()),
                "Recall": float(metrics.recall.mean()),
                "F1": float(metrics.f1.mean()),
                "Seed(s)": ",".join(map(str, seeds)),
                "Status": source + "; historical leakage retained",
            })
    return pd.DataFrame(table8), pd.DataFrame(table9)


def build(run_root, output_dir):
    run_root, output_dir = Path(run_root).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    x, _, _, _, splits, expected = aligned_data()
    expected_positions = np.concatenate([x.index[test].to_numpy() for _, test in splits])
    seed42_8, seed42_9 = make_tables(run_root, (42,), expected, expected_positions)
    mean_8, mean_9 = make_tables(run_root, (42, 43, 44), expected, expected_positions)
    pairs = {seed: (run_root / f"seed{seed}" / "full" / "predictions.csv",
                    run_root / f"seed{seed}" / "no_fcnn" / "predictions.csv")
             for seed in (42, 43, 44)}
    dm = save_dm_tables(pairs, REFERENCE, output_dir / "dm_audit")
    sheets = {
        "Table8_seed42": seed42_8,
        "Table9_seed42": seed42_9,
        "DM_seed42": dm["seed42"],
        "Table8_mean_42_43_44": mean_8,
        "Table9_mean_42_43_44": mean_9,
        "DM_mean_42_43_44": dm["three_seed"],
    }
    for name, frame in sheets.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False,
                     encoding="utf-8-sig", float_format="%.17g")
    with pd.ExcelWriter(output_dir / "six_tables.xlsx", engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    (output_dir / "README.md").write_text(
        "# New aligned rerun tables\n\n"
        "These six tables are new results, not replacements for historical Table 8/9. "
        "Random Walk, CNN-LSTM regression, and without-FE are explicitly reconstructed baselines. "
        "Table 8/9 use the first unique 3,680 test predictions per seed and mean of five "
        "fold metrics. Seed-42 Table 8 standard deviations describe five folds; the "
        "three-seed Table 8 standard deviations describe the three five-fold seed means. "
        "Table 9 reports the mean of three five-fold seed means. "
        "DM compares paired first-round full versus zero-FCNN daily losses, with Bartlett "
        "HAC lag 10 and a two-sided modified DM test. Three-seed DM averages each date's "
        "paired losses before testing; p-values are never averaged. Historical classification "
        "labels and meta-evaluation leakage remain, so these are not leakage-free next-day scores.\n",
        encoding="utf-8")
    return sheets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.runs, args.output)
