"""BFNE-Net without feature engineering, rebuilt from the paper (not the authors' script).

Uses only the 11 raw variables in Table 1 and drops all 37 engineered features from
Table 2. Everything else is unchanged: two FCNNs + three trees -> GBM, same labels,
same five folds. One run takes a long time because each of the 10 training folds
redoes the Optuna search."""

import argparse
import importlib.util
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from historical_alignment import ROOT, aligned_data
from BFNE_Net_without_FCNNs import prepare_fold, fit_meta_models
from regression_scale_utils import regression_metrics_original_price


RAW_FEATURES = (
    "CrudeOil_SpotPrice_BrentUK", "China_CPI_YoY_CurrentMonth", "US_CPI_YoY_NSA",
    "Japan_CPI_YoY", "SaudiArabia_CPI_YoY", "Turkey_CPI_YoY",
    "US_DowJones_IndustrialAverage", "US_SP500_Index", "SpotRate_USD_CNY",
    "Interbank_OpenRate_USD_INR", "SpotRate_Tokyo_9AM_USD_JPY",
)


def audit_features(x):
    if len(RAW_FEATURES) != 11 or len(set(RAW_FEATURES)) != 11:
        raise ValueError("Table 1 raw feature list is incomplete")
    missing = set(RAW_FEATURES) - set(x.columns)
    if missing:
        raise ValueError(f"Missing Table 1 inputs: {sorted(missing)}")
    engineered = [column for column in x.columns if column not in RAW_FEATURES]
    if len(engineered) != 37 or len(x.columns) != 48:
        raise ValueError("Unexpected engineered feature count")
    selected = x.loc[:, list(RAW_FEATURES)]
    if any(column in engineered for column in selected):
        raise ValueError("Engineered input remains in without-FE model")
    return selected, engineered


def load_full_model_source():
    path = ROOT / "BFNE_Net.py"
    spec = importlib.util.spec_from_file_location("full_bfne_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_fold(base, fd, device):
    reg_parts = base.train_base_models_reg(fd["Xr_train"], fd["yr_train_fit"], device)
    reg_models, reg_opt, reg_sch, reg_stop, reg_loss = reg_parts
    reg_loader = DataLoader(base.FinancialRegressionDataset(
        fd["Xr_train"], fd["yr_train_fit"]), batch_size=32, shuffle=True)
    reg_test = DataLoader(base.FinancialRegressionDataset(
        fd["Xr_test"], fd["yr_test_fit"]), batch_size=32, shuffle=False)
    base.train_fcnn_reg(reg_models, reg_opt, reg_sch, reg_stop, reg_loss,
                        reg_loader, device, num_epochs=300)
    for model in reg_models[2:]:
        model.fit(fd["Xr_train"], fd["yr_train_fit"])
    reg_features = base.generate_meta_features_reg(
        *reg_models, reg_test, fd["Xr_test"], device)

    clf_parts = base.train_base_models_clf(fd["Xc_train"], fd["yc_train"], device)
    clf_models, clf_opt, clf_sch, clf_stop, clf_loss = clf_parts
    clf_loader = DataLoader(base.FinancialClassificationDataset(
        fd["Xc_train"], fd["yc_train"]), batch_size=32, shuffle=True)
    clf_test = DataLoader(base.FinancialClassificationDataset(
        fd["Xc_test"], fd["yc_test"]), batch_size=32, shuffle=False)
    base.train_fcnn_clf(clf_models, clf_opt, clf_sch, clf_stop, clf_loss,
                        clf_loader, device, num_epochs=300)
    for model in clf_models[2:]:
        model.fit(fd["Xc_train"], fd["yc_train"])
    clf_features = base.generate_meta_features_clf(
        *clf_models, clf_test, fd["Xc_test"], device)
    if reg_features.shape[1] != 5 or clf_features.shape[1] != 10:
        raise ValueError("The full two-FCNN stacking dimensions changed")
    return reg_features, clf_features


def main():
    """Run the without-FE model with the forward-chaining protocol in run_seeded_ensembles.py."""
    import argparse
    from run_seeded_ensembles import run

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run("without_fe", args.seed, args.output)


if __name__ == "__main__":
    main()
