"""CNN-LSTM.

Classification reuses lstm/CNN_LSTM_tuned.py (48 features, inner learning-rate pick
in time order, five outer folds) on the same 4,419 rows BFNE uses. The regression
head is new: a single output added on top. The authors' regression version wasn't
in their project."""

import argparse
import importlib.util
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from historical_alignment import ROOT, aligned_data
from regression_scale_utils import regression_metrics_original_price


TUNED_SOURCE = ROOT / "lstm" / "CNN_LSTM_tuned.py"


def load_tuned_module(output_dir):
    spec = importlib.util.spec_from_file_location("historical_tuned_cnn_lstm", TUNED_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUTPUT_DIR = output_dir
    module.PICTURE_DIR = output_dir / "picture"
    return module


def run_regression(module, x, y_price, expected, splits, device, output_dir, seed=42):
    """Train the new single-output head separately on each training fold."""
    rows, metrics = [], []
    for fold, (train, test) in enumerate(splits, 1):
        module.set_seed(seed)
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        x_train = scaler_x.fit_transform(x.iloc[train]).astype(np.float32)
        x_test = scaler_x.transform(x.iloc[test]).astype(np.float32)
        y_train = scaler_y.fit_transform(
            y_price.iloc[train].to_numpy().reshape(-1, 1)).astype(np.float32)
        dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(dataset, batch_size=module.BATCH_SIZE, shuffle=True,
                            generator=generator, num_workers=0)
        model = module.CNNLSTM(hidden_size=128, num_classes=1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        loss_fn = torch.nn.MSELoss()
        for epoch in range(100):
            model.train()
            for features, target in loader:
                features, target = features.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(features), target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            print(f"Regression fold {fold}/5 epoch {epoch + 1}/100", flush=True)
        model.eval()
        with torch.no_grad():
            pred_fit = np.concatenate([
                model(torch.from_numpy(batch).to(device)).cpu().numpy().ravel()
                for batch in np.array_split(x_test, max(1, int(np.ceil(len(x_test) / 32))))
            ])
        truth = y_price.iloc[test].to_numpy(dtype=float)
        pred_price, rmse, mape = regression_metrics_original_price(
            truth, pred_fit, prediction_scaler=scaler_y)
        frame = expected.loc[expected.fold.eq(fold)].copy()
        if not np.allclose(frame.y_true_price, truth, rtol=0, atol=1e-8):
            raise ValueError("Regression truth does not match historical fold")
        frame["row_index"] = x.index[test].to_numpy()
        frame["y_pred_price_original"] = pred_price
        frame["y_pred_price_scaled"] = pred_fit
        rows.append(frame)
        metrics.append({"fold": fold, "rmse_usd_per_oz": rmse,
                        "mape_percent": mape})
        torch.save(model.state_dict(), output_dir / f"regression_model_fold{fold}.pt")
        (output_dir / f"regression_scaler_y_fold{fold}.json").write_text(
            json.dumps({"mean": scaler_y.mean_.tolist(), "scale": scaler_y.scale_.tolist()}),
            encoding="utf-8")
    pd.concat(rows, ignore_index=True).to_csv(
        output_dir / "regression_predictions.csv", index=False,
        encoding="utf-8-sig", float_format="%.17g")
    pd.DataFrame(metrics).to_csv(output_dir / "regression_fold_metrics.csv", index=False,
                                  encoding="utf-8-sig", float_format="%.17g")


def run(output_dir, task="both", seed=42):
    output_dir = Path(output_dir)
    x, y_class, y_price, dates, splits, expected = aligned_data()
    output_dir.mkdir(parents=True, exist_ok=False)
    module = load_tuned_module(output_dir)
    original_set_seed = module.set_seed
    original_make_loader = module.make_loader
    module.set_seed = lambda selected_seed=None: original_set_seed(
        seed if selected_seed is None else selected_seed)
    module.make_loader = lambda x_batch, y_batch, shuffle, selected_seed=None: original_make_loader(
        x_batch, y_batch, shuffle, seed if selected_seed is None else selected_seed)
    module.PICTURE_DIR.mkdir(parents=True)
    module.set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = {
        "status": "running", "source_classification": str(TUNED_SOURCE),
        "classification_change": "same tuned model, aligned to historical 4,419-row population",
        "regression_change": "new one-output price head and fold-specific y StandardScaler; reconstructed baseline",
        "author_regression_implementation_recovered": False,
        "seed": seed, "device": str(device), "task": task,
        "n_cleaned_rows": len(x), "n_expected_test_rows": len(expected),
        "classification_label": "(GOLD.diff() > 0)",
        "regression_label": "GOLD.shift(-1)",
        "leakage_note": "Historical classification label leaks via gold-derived inputs.",
    }
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        if task in ("classification", "both"):
            selected, tuning = module.tune_learning_rate(x, y_class, device)
            folds, losses, predictions, _ = module.run_cross_validation(
                x, y_class, selected, device)
            expected_positions = np.concatenate([test for _, test in splits])
            if len(predictions) != 3680 or not np.array_equal(
                    predictions.y_true.to_numpy(), expected.y_true_class.to_numpy()) or not np.array_equal(
                    predictions.row_index.to_numpy(), expected_positions) or not np.array_equal(
                    predictions.fold.to_numpy(), expected.fold.to_numpy()):
                raise ValueError("CNN-LSTM fold positions or classification targets are misaligned")
            predictions = predictions.reset_index(drop=True)
            predictions["sample_position"] = predictions.row_index.to_numpy()
            predictions["row_index"] = x.index[expected_positions].to_numpy()
            predictions["date"] = expected.date.to_numpy()
            predictions["y_true_price"] = expected.y_true_price.to_numpy()
            predictions.to_csv(output_dir / "classification_predictions.csv", index=False,
                               encoding="utf-8-sig", float_format="%.17g")
            folds.to_csv(output_dir / "classification_fold_metrics.csv", index=False,
                         encoding="utf-8-sig", float_format="%.17g")
            tuning.to_csv(output_dir / "classification_tuning.csv", index=False,
                          encoding="utf-8-sig", float_format="%.17g")
            losses.to_csv(output_dir / "classification_epoch_losses.csv", index=False,
                          encoding="utf-8-sig", float_format="%.17g")
            manifest["selected_classification_parameters"] = {
                "optimizer": selected["optimizer"], "learning_rate": float(selected["lr"])}
        if task in ("regression", "both"):
            run_regression(module, x, y_price, expected, splits, device, output_dir, seed)
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        raise
    finally:
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("classification", "regression", "both"),
                        default="both")
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "reconstructed_runs" /
                        f"cnn_lstm_{datetime.now():%Y%m%d_%H%M%S}")
    args = parser.parse_args()
    print(run(args.output, task=args.task, seed=args.seed))
