"""Rerun the standalone tree models and the random walk on the aligned dates."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from historical_alignment import DATA, aligned_data
from regression_scale_utils import regression_metrics_original_price
from run_seeded_ensembles import class_metrics, record


def estimators(model, seed):
    if model == "random_forest":
        params = dict(n_estimators=200, max_depth=10, random_state=seed, n_jobs=-1)
        return RandomForestRegressor(**params), RandomForestClassifier(**params)
    if model == "xgboost":
        params = dict(n_estimators=300, learning_rate=0.03, max_depth=7,
                      subsample=0.9, colsample_bytree=0.9, random_state=seed, n_jobs=-1)
        return (XGBRegressor(**params, objective="reg:squarederror"),
                XGBClassifier(**params, objective="binary:logistic",
                              use_label_encoder=False, eval_metric="logloss"))
    if model == "lightgbm":
        params = dict(n_estimators=500, learning_rate=0.01, max_depth=15,
                      num_leaves=100, subsample=0.5, colsample_bytree=0.5,
                      random_state=seed)
        return LGBMRegressor(**params), LGBMClassifier(**params)
    raise ValueError(model)


def run(model, seed, output_dir):
    if model not in ("random_forest", "xgboost", "lightgbm", "random_walk"):
        raise ValueError(model)
    if seed not in (42, 43, 44):
        raise ValueError(seed)
    x, y_class, y_price, _, splits, expected = aligned_data()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest = dict(status="running", model=model, seed=seed, n_cleaned_rows=4419,
                    n_test_predictions=3680, data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
                    regression_target="GOLD.shift(-1)", classification_target="(GOLD.diff() > 0)",
                    regression_unit="USD/oz", mape_unit="%",
                    persistence_reconstruction=model == "random_walk",
                    original_author_implementation=model != "random_walk",
                    historical_leakage_note="Historical classification label remains leaky.")
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(old[k] != manifest[k] for k in ("model", "seed", "data_sha256")):
            raise ValueError("Cannot resume different experiment")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        gold_today = pd.read_excel(DATA).loc[x.index, "GOLD"].to_numpy(dtype=float)
        for fold, (train, test) in enumerate(splits, 1):
            path = output_dir / f"fold{fold}_predictions.csv"
            if path.exists():
                continue
            print(f"{model} seed {seed} fold {fold}/5", flush=True)
            frame = expected.loc[expected.fold.eq(fold)].copy().reset_index(drop=True)
            frame["row_index"] = x.index[test].to_numpy()
            frame["y_true_price_original"] = frame.pop("y_true_price")
            if model == "random_walk":
                frame["y_pred_price_original"] = gold_today[test]
            else:
                scaler = StandardScaler()
                train_x = scaler.fit_transform(x.iloc[train])
                test_x = scaler.transform(x.iloc[test])
                reg, clf = estimators(model, seed)
                reg.fit(train_x, y_price.iloc[train])
                clf.fit(train_x, y_class.iloc[train])
                frame["y_pred_price_original"] = reg.predict(test_x)
                frame["y_pred_class"] = clf.predict(test_x)
                frame["y_pred_probability"] = clf.predict_proba(test_x)[:, 1]
            record(path, frame)
        frames = [pd.read_csv(output_dir / f"fold{i}_predictions.csv", float_precision="round_trip")
                  for i in range(1, 6)]
        combined = pd.concat(frames, ignore_index=True)
        if len(combined) != 3680 or not np.array_equal(combined.date, expected.date):
            raise ValueError("Test dates differ from historical reference")
        record(output_dir / "predictions.csv", combined)
        rows = []
        for frame in frames:
            _, rmse, mape = regression_metrics_original_price(frame.y_true_price_original,
                                                                 frame.y_pred_price_original)
            row = dict(fold=int(frame.fold.iat[0]), rmse_usd_per_oz=rmse, mape_percent=mape)
            if model != "random_walk":
                row.update(class_metrics(frame.y_true_class, frame.y_pred_class,
                                         frame.y_pred_probability))
            rows.append(row)
        record(output_dir / "fold_metrics.csv", pd.DataFrame(rows))
        manifest["status"] = "completed"
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("random_forest", "xgboost", "lightgbm", "random_walk"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.model, args.seed, args.output)
