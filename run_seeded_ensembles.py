"""Seeded runs of the BFNE stacking models, checkpointed per fold.

Base models for fold k are trained on the data before fold k and predict fold k. The meta
model used to score fold k is trained only on the out-of-fold predictions of the earlier
folds, so it never sees a label from the block it predicts. Fold 1 therefore only feeds
the first meta model and the models are scored on folds 2-5. A resumed run uses the same
per-fold seed."""

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import BFNE_Net_without_FCNNs as zero
from historical_alignment import DATA, ROOT, aligned_data
from protocol import EVAL_FOLDS, FOLD_SIZE
from BFNE_Net_without_FE import audit_features, load_full_model_source, train_fold as full_train_fold
from regression_scale_utils import regression_metrics_original_price


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seeded_optimizers(module, seed):
    """Same Optuna objectives and search space, 50 trials per search."""
    def make_optimizer(task):
        objective = getattr(module, f"objective_lgbm_{task}")
        direction = "minimize" if task == "reg" else "maximize"

        def optimize(x_train, y_train):
            slot = getattr(module, "_search_slot", None)
            if slot is None:
                raise RuntimeError("Set the fold search slot before Optuna")
            sampler = optuna.samplers.TPESampler(seed=seed * 10000 + slot)
            study = optuna.create_study(direction=direction, sampler=sampler)
            study.optimize(lambda trial: objective(trial, x_train, y_train), n_trials=50)
            return study.best_params
        return optimize

    module.optimize_lgbm_reg = make_optimizer("reg")
    module.optimize_lgbm_clf = make_optimizer("clf")


def configure_module(module, model, seed):
    module._search_slot = None
    seeded_optimizers(module, seed)
    if model == "no_fcnn":
        original_reg = module.build_regressors
        original_clf = module.build_classifiers

        def build_reg(*args):
            models = original_reg(*args)
            for estimator in models:
                estimator.set_params(random_state=seed)
            return models

        def build_clf(*args):
            models = original_clf(*args)
            for estimator in models:
                estimator.set_params(random_state=seed)
            return models

        module.build_regressors = build_reg
        module.build_classifiers = build_clf
    else:
        for task in ("reg", "clf"):
            original = getattr(module, f"train_base_models_{task}")

            def make_wrapper(function):
                def wrapper(*args):
                    parts = function(*args)
                    for estimator in parts[0][2:]:
                        estimator.set_params(random_state=seed)
                    return parts
                return wrapper

            setattr(module, f"train_base_models_{task}", make_wrapper(original))


def meta_models(reg_features, reg_labels, clf_features, clf_labels, seed):
    reg = GradientBoostingRegressor(n_estimators=300, learning_rate=0.15,
                                    max_depth=5, random_state=seed)
    clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.15,
                                     max_depth=5, random_state=seed)
    reg.fit(np.vstack(reg_features), np.asarray(reg_labels))
    clf.fit(np.vstack(clf_features), np.asarray(clf_labels))
    return reg, clf


def forward_meta_models(blocks, fold, seed):
    """Meta models for `fold`, fit only on the out-of-fold blocks that come before it."""
    earlier = [b for b in blocks if b["fold"] < fold]
    if not earlier:
        raise ValueError("The first block has no earlier data to train a meta model on")
    return meta_models([b["reg"] for b in earlier], np.concatenate([b["reg_labels"] for b in earlier]),
                       [b["clf"] for b in earlier], np.concatenate([b["clf_labels"] for b in earlier]),
                       seed)


def class_metrics(y, pred, probability):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, probability)),
    }


def record(path, frame):
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")


def run(model, seed, output_dir):
    if model not in ("full", "no_fcnn", "without_fe") or seed not in (42, 43, 44):
        raise ValueError("Unsupported model or seed")
    output_dir = Path(output_dir).resolve()
    x_all, y_class, y_price, _, splits, expected = aligned_data()
    x, deleted = audit_features(x_all) if model == "without_fe" else (x_all, [])
    source = zero if model == "no_fcnn" else load_full_model_source()
    configure_module(source, model, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = output_dir / "run_manifest.json"
    manifest = {
        "status": "running", "model": model, "seed": seed,
        "source": "BFNE_Net_without_FCNNs.py" if model == "no_fcnn" else "BFNE_Net.py",
        "paper_reconstruction": model == "without_fe",
        "n_cleaned_rows": 4419, "n_features": x.shape[1],
        "deleted_engineered_features": deleted,
        "outer_folds": 5, "evaluation_folds": list(EVAL_FOLDS),
        "optuna_trials_per_fold_per_task": 50,
        "optuna_objective_cv": 3,
        "optuna_objective_model_seed": 42,
        "optuna_sampler_seed_rule": "seed*10000 + chronological_fold_slot (same slot for regression and classification)",
        "classification_label": "(GOLD.shift(-1) > GOLD)",
        "regression_label": "GOLD.shift(-1)",
        "optuna_inner_cv": "TimeSeriesSplit(3)",
        "regression_unit": "USD/oz", "mape_unit": "%",
        "protocol": "meta model for fold k trained on out-of-fold blocks before k only",
        "device": str(device),
        "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    if manifest_file.exists():
        existing = json.loads(manifest_file.read_text(encoding="utf-8"))
        if existing["model"] != model or existing["seed"] != seed or existing["data_sha256"] != manifest["data_sha256"]:
            raise ValueError("Cannot resume a different model, seed, or dataset")
        manifest = existing
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    train_fold = source.train_fold if model == "no_fcnn" else lambda fd: full_train_fold(source, fd, device)
    try:
        blocks = []
        for fold, (train, test) in enumerate(splits, 1):
            fd = zero.prepare_fold(x, y_class, y_price, train, test)
            checkpoint = output_dir / f"oof_fold{fold}.npz"
            if checkpoint.exists():
                saved = np.load(checkpoint)
                reg_meta, clf_meta = saved["reg"], saved["clf"]
            else:
                seed_everything(seed + fold)
                source._search_slot = fold
                print(f"{model} seed {seed}: base models, fold {fold}/5", flush=True)
                reg_meta, clf_meta = train_fold(fd)
                np.savez_compressed(checkpoint, reg=reg_meta, clf=clf_meta)
            # every fold has its own y scaler, so put the regression meta-features and labels
            # back in USD/oz before stacking; the meta model then sees one common scale
            scaler = fd["scaler_y"]
            blocks.append(dict(fold=fold, test=test, fd=fd,
                               reg=reg_meta * scaler.scale_[0] + scaler.mean_[0], clf=clf_meta,
                               reg_labels=np.asarray(fd["yr_test"], dtype=float),
                               clf_labels=fd["yc_test"].to_numpy()))
        frames, rows = [], []
        for block in blocks:
            fold, fd = block["fold"], block["fd"]
            if fold not in EVAL_FOLDS:
                continue
            meta_reg, meta_clf = forward_meta_models(blocks, fold, seed)
            pred_price, rmse, mape = regression_metrics_original_price(
                fd["yr_test"], meta_reg.predict(block["reg"]))
            pred_class = meta_clf.predict(block["clf"])
            probability = meta_clf.predict_proba(block["clf"])[:, 1]
            frame = expected.loc[expected.fold.eq(fold)].copy().reset_index(drop=True)
            if not np.array_equal(frame.y_true_class, fd["yc_test"].to_numpy()) or not np.allclose(
                    frame.y_true_price, fd["yr_test"], rtol=0, atol=1e-8):
                raise ValueError("Test truth mismatch")
            frame["row_index"] = x.index[block["test"]].to_numpy()
            frame["y_true_price_original"] = frame.pop("y_true_price")
            frame["y_pred_price_original"] = pred_price
            frame["y_pred_class"] = pred_class
            frame["y_pred_probability"] = probability
            record(output_dir / f"fold{fold}_predictions.csv", frame)
            (output_dir / f"scaler_y_fold{fold}.json").write_text(
                json.dumps({"mean": fd["scaler_y"].mean_.tolist(),
                            "scale": fd["scaler_y"].scale_.tolist()}), encoding="utf-8")
            rows.append({"fold": fold, "rmse_usd_per_oz": rmse, "mape_percent": mape,
                         **class_metrics(frame.y_true_class, pred_class, probability)})
            frames.append(frame)
            print(f"{model} seed {seed} fold {fold}: RMSE USD/oz={rmse:.4f}; MAPE %={mape:.4f}", flush=True)
        predictions = pd.concat(frames, ignore_index=True)
        if len(predictions) != len(EVAL_FOLDS) * FOLD_SIZE:
            raise ValueError("Unexpected number of test predictions")
        record(output_dir / "predictions.csv", predictions)
        record(output_dir / "fold_metrics.csv", pd.DataFrame(rows))
        manifest.update(status="completed", n_test_predictions=len(predictions),
                        completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("full", "no_fcnn", "without_fe"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.model, args.seed, args.output)
