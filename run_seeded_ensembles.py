"""Auditable seeded reruns of the historical BFNE stacking protocols.

The historical labels and meta-evaluation leakage are deliberately retained.
Each completed fold is checkpointed; a resumed run uses the same per-fold seed.
"""

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

import no_fcnn_model as zero
from historical_alignment import DATA, ROOT, aligned_data
from reconstructed_without_fe import audit_features, load_full_model_source, train_fold as full_train_fold
from regression_scale_utils import regression_metrics_original_price


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seeded_optimizers(module, seed):
    """Keep original Optuna objectives/search spaces and 50-trial budgets."""
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


def run(model, seed, output_dir, evaluation_rounds=5):
    if model not in ("full", "no_fcnn", "without_fe") or seed not in (42, 43, 44):
        raise ValueError("Unsupported model or seed")
    output_dir = Path(output_dir).resolve()
    x_all, y_class, y_price, _, splits, expected = aligned_data()
    x, deleted = audit_features(x_all) if model == "without_fe" else (x_all, [])
    source = zero if model == "no_fcnn" else load_full_model_source()
    configure_module(source, model, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rounds = 1 if model == "without_fe" else evaluation_rounds
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = output_dir / "run_manifest.json"
    manifest = {
        "status": "running", "model": model, "seed": seed,
        "source": "no_fcnn_model.py" if model == "no_fcnn" else "归一.py",
        "paper_reconstruction": model == "without_fe",
        "n_cleaned_rows": 4419, "n_features": x.shape[1],
        "deleted_engineered_features": deleted,
        "outer_folds": 5, "evaluation_rounds": rounds,
        "optuna_trials_per_fold_per_task": 50,
        "optuna_objective_cv": 3,
        "optuna_objective_model_seed": 42,
        "optuna_sampler_seed_rule": "seed*10000 + chronological_fold_slot (same slot for regression and classification)",
        "first_round_test_predictions": 3680,
        "classification_label": "(GOLD.diff() > 0)",
        "regression_label": "GOLD.shift(-1)",
        "regression_unit": "USD/oz", "mape_unit": "%",
        "leakage_note": "Historical classification-label and stacking evaluation leakage remain.",
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
        reg_parts, reg_labels, clf_parts, clf_labels = [], [], [], []
        for fold, (train, test) in enumerate(splits, 1):
            checkpoint = output_dir / f"oof_fold{fold}.npz"
            if checkpoint.exists():
                saved = np.load(checkpoint)
                reg_meta, clf_meta = saved["reg"], saved["clf"]
            else:
                seed_everything(seed + fold)
                source._search_slot = fold
                fd = zero.prepare_fold(x, y_class, y_price, train, test)
                print(f"{model} seed {seed}: OOF fold {fold}/5", flush=True)
                reg_meta, clf_meta = train_fold(fd)
                np.savez_compressed(checkpoint, reg=reg_meta, clf=clf_meta)
            fd = zero.prepare_fold(x, y_class, y_price, train, test)
            reg_parts.append(reg_meta)
            reg_labels.extend(fd["yr_test_fit"])
            clf_parts.append(clf_meta)
            clf_labels.extend(fd["yc_test"].to_numpy())
        meta_reg, meta_clf = meta_models(reg_parts, reg_labels, clf_parts, clf_labels, seed)
        for round_number in range(1, rounds + 1):
            for fold, (train, test) in enumerate(splits, 1):
                path = output_dir / f"round{round_number}_fold{fold}_predictions.csv"
                if path.exists():
                    continue
                seed_everything(seed + 100 * round_number + fold)
                source._search_slot = 5 + 5 * (round_number - 1) + fold
                fd = zero.prepare_fold(x, y_class, y_price, train, test)
                print(f"{model} seed {seed}: evaluation round {round_number}/{rounds}, fold {fold}/5", flush=True)
                reg_meta, clf_meta = train_fold(fd)
                pred_fit = meta_reg.predict(reg_meta)
                pred_price, rmse, mape = regression_metrics_original_price(
                    fd["yr_test"], pred_fit, prediction_scaler=fd["scaler_y"])
                pred_class = meta_clf.predict(clf_meta)
                probability = meta_clf.predict_proba(clf_meta)[:, 1]
                frame = expected.loc[expected.fold.eq(fold)].copy().reset_index(drop=True)
                if not np.array_equal(frame.y_true_class, fd["yc_test"].to_numpy()) or not np.allclose(
                        frame.y_true_price, fd["yr_test"], rtol=0, atol=1e-8):
                    raise ValueError("Historical truth mismatch")
                frame.insert(0, "round", round_number)
                frame["row_index"] = x.index[test].to_numpy()
                frame["y_true_price_original"] = frame.pop("y_true_price")
                frame["y_pred_price_original"] = pred_price
                frame["y_pred_price_scaled"] = pred_fit
                frame["y_pred_class"] = pred_class
                frame["y_pred_probability"] = probability
                record(path, frame)
                (output_dir / f"round{round_number}_scaler_y_fold{fold}.json").write_text(
                    json.dumps({"mean": fd["scaler_y"].mean_.tolist(),
                                "scale": fd["scaler_y"].scale_.tolist()}), encoding="utf-8")
                print(f"RMSE USD/oz={rmse:.4f}; MAPE %={mape:.4f}", flush=True)
            # The historical scripts refit the unchanged OOF meta models
            # between evaluation rounds. This is deterministic for each seed.
            if round_number < rounds:
                meta_reg, meta_clf = meta_models(
                    reg_parts, reg_labels, clf_parts, clf_labels, seed)
        all_frames = [pd.read_csv(output_dir / f"round{round_number}_fold{fold}_predictions.csv",
                                  float_precision="round_trip")
                      for round_number in range(1, rounds + 1) for fold in range(1, 6)]
        first = pd.concat(all_frames[:5], ignore_index=True)
        if len(first) != 3680 or first.groupby("fold").size().to_dict() != dict.fromkeys(range(1, 6), 736):
            raise ValueError("First-round predictions are not 5 x 736")
        record(output_dir / "predictions.csv", first)
        record(output_dir / "all_round_predictions.csv", pd.concat(all_frames, ignore_index=True))
        rows = []
        for frame in all_frames:
            _, rmse, mape = regression_metrics_original_price(frame.y_true_price_original,
                                                                 frame.y_pred_price_original)
            row = {"round": int(frame["round"].iat[0]), "fold": int(frame.fold.iat[0]),
                   "rmse_usd_per_oz": rmse, "mape_percent": mape}
            if row["round"] == 1:
                row.update(class_metrics(frame.y_true_class, frame.y_pred_class,
                                         frame.y_pred_probability))
            rows.append(row)
        record(output_dir / "fold_metrics.csv", pd.DataFrame(rows))
        manifest.update(status="completed", n_first_round_predictions=3680,
                        n_all_round_predictions=sum(len(f) for f in all_frames),
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
    parser.add_argument("--rounds", type=int, default=5, choices=(1, 2, 3, 4, 5),
                        help="Evaluation rounds; tables use round 1 only")
    args = parser.parse_args()
    run(args.model, args.seed, args.output, args.rounds)
