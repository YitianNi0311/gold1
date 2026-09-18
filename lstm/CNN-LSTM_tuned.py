from __future__ import annotations

import importlib.util
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 100
TUNING_EPOCHS = 30
TUNING_PATIENCE = 8
N_SPLITS = 5
WEIGHT_DECAY = 1e-5

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_PATH = PROJECT_DIR / "GOLD_cleaned.xlsx"
ORIGINAL_SCRIPT = PROJECT_DIR / "CNN-LSTM.py"
OUTPUT_DIR = SCRIPT_DIR
PICTURE_DIR = OUTPUT_DIR / "picture"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_original_preprocessor():
    spec = importlib.util.spec_from_file_location("original_cnn_lstm", ORIGINAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {ORIGINAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.preprocess_data


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


class CNNLSTM(nn.Module):
    """Original CNN/LSTM body with the missing two-class output layer restored."""

    def __init__(self, hidden_size: int = 128, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=0.3,
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        return self.classifier(x)


def make_loader(x, y, shuffle: bool, seed: int = SEED) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ArrayDataset(np.asarray(x), np.asarray(y)),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    count = 0
    labels, predictions, probabilities = [], [], []
    with torch.no_grad():
        for features, target in loader:
            features, target = features.to(device), target.to(device)
            logits = model(features)
            loss = criterion(logits, target)
            probability = torch.softmax(logits, dim=1)[:, 1]
            total_loss += loss.item() * len(target)
            count += len(target)
            labels.extend(target.cpu().numpy().tolist())
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            probabilities.extend(probability.cpu().numpy().tolist())
    auc = roc_auc_score(labels, probabilities)
    return total_loss / count, auc, np.array(labels), np.array(predictions), np.array(probabilities)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    count = 0
    for features, target in loader:
        features, target = features.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(target)
        count += len(target)
    return total_loss / count


def tune_learning_rate(x: pd.DataFrame, y: pd.Series, device):
    first_outer_train, _ = next(TimeSeriesSplit(n_splits=N_SPLITS).split(x))
    split_at = int(len(first_outer_train) * 0.8)
    inner_train = first_outer_train[:split_at]
    inner_val = first_outer_train[split_at:]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x.iloc[inner_train])
    x_val = scaler.transform(x.iloc[inner_val])
    y_train = y.iloc[inner_train].to_numpy()
    y_val = y.iloc[inner_val].to_numpy()

    candidates = [
        {"name": "original_optimizer_higher_lr", "optimizer": "Adam", "lr": 5e-4},
        {"name": "reference_guiyi", "optimizer": "AdamW", "lr": 5e-4},
        {"name": "adamw_lr_1e-3", "optimizer": "AdamW", "lr": 1e-3},
        {"name": "adamw_lr_2e-4", "optimizer": "AdamW", "lr": 2e-4},
    ]
    rows = []
    for candidate in candidates:
        set_seed()
        model = CNNLSTM().to(device)
        criterion = nn.CrossEntropyLoss()
        if candidate["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=candidate["lr"], weight_decay=WEIGHT_DECAY)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=candidate["lr"])
        train_loader = make_loader(x_train, y_train, True)
        val_loader = make_loader(x_val, y_val, False)
        best_auc = -np.inf
        best_loss = np.inf
        best_epoch = 0
        stale = 0
        started = time.time()
        for epoch in range(1, TUNING_EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_auc, *_ = evaluate(model, val_loader, criterion, device)
            if val_auc > best_auc + 1e-4:
                best_auc, best_loss, best_epoch, stale = val_auc, val_loss, epoch, 0
            else:
                stale += 1
            print(
                f"TUNE {candidate['name']} epoch {epoch:03d}: "
                f"loss={train_loss:.6f} val_loss={val_loss:.6f} val_auc={val_auc:.6f}",
                flush=True,
            )
            if stale >= TUNING_PATIENCE:
                break
        rows.append({
            **candidate,
            "best_inner_val_auc": best_auc,
            "best_inner_val_loss": best_loss,
            "best_epoch": best_epoch,
            "epochs_run": epoch,
            "seconds": time.time() - started,
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tuning = pd.DataFrame(rows).sort_values(
        ["best_inner_val_auc", "best_inner_val_loss"], ascending=[False, True]
    ).reset_index(drop=True)
    return tuning.iloc[0].to_dict(), tuning


def run_cross_validation(x, y, selected, device):
    criterion = nn.CrossEntropyLoss()
    fold_rows, loss_rows, prediction_frames = [], [], []
    roc_data = []

    for fold, (train_index, test_index) in enumerate(TimeSeriesSplit(n_splits=N_SPLITS).split(x), start=1):
        set_seed()
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x.iloc[train_index])
        x_test = scaler.transform(x.iloc[test_index])
        y_train = y.iloc[train_index].to_numpy()
        y_test = y.iloc[test_index].to_numpy()
        train_loader = make_loader(x_train, y_train, True)
        test_loader = make_loader(x_test, y_test, False)

        model = CNNLSTM().to(device)
        if selected["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(selected["lr"]), weight_decay=WEIGHT_DECAY)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=float(selected["lr"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
        started = time.time()
        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_auc, *_ = evaluate(model, test_loader, criterion, device)
            current_lr = optimizer.param_groups[0]["lr"]
            loss_rows.append({
                "fold": fold,
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss_monitor_only": test_loss,
                "test_auc_monitor_only": test_auc,
                "learning_rate": current_lr,
            })
            print(
                f"FINAL fold {fold} epoch {epoch:03d}/{MAX_EPOCHS}: "
                f"loss={train_loss:.6f} test_loss={test_loss:.6f} test_auc={test_auc:.6f}",
                flush=True,
            )
            scheduler.step()

        test_loss, auc, labels, predictions, probabilities = evaluate(model, test_loader, criterion, device)
        fold_rows.append({
            "fold": fold,
            "train_samples": len(train_index),
            "test_samples": len(test_index),
            "positive_rate_train": float(np.mean(y_train)),
            "positive_rate_test": float(np.mean(y_test)),
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, zero_division=0),
            "recall": recall_score(labels, predictions, zero_division=0),
            "f1": f1_score(labels, predictions, zero_division=0),
            "auc": auc,
            "final_test_loss": test_loss,
            "seconds": time.time() - started,
        })
        prediction_frames.append(pd.DataFrame({
            "row_index": test_index,
            "fold": fold,
            "y_true": labels,
            "y_pred": predictions,
            "probability_class_1": probabilities,
        }))
        fpr, tpr, _ = roc_curve(labels, probabilities)
        roc_data.append((fold, fpr, tpr, auc))
        torch.save(model.state_dict(), OUTPUT_DIR / f"model_fold_{fold}.pt")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(fold_rows), pd.DataFrame(loss_rows), pd.concat(prediction_frames), roc_data


def save_plots(losses, roc_data):
    plt.figure(figsize=(10, 6))
    for fold, frame in losses.groupby("fold"):
        plt.plot(frame["epoch"], frame["train_loss"], label=f"Fold {fold}")
    plt.xlabel("Epoch")
    plt.ylabel("Training cross-entropy loss")
    plt.title("CNN-LSTM training loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(PICTURE_DIR / "training_loss.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 7))
    for fold, fpr, tpr, auc in roc_data:
        plt.plot(fpr, tpr, label=f"Fold {fold} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.500)")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("CNN-LSTM ROC curves")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(PICTURE_DIR / "roc_curves.png", dpi=200)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; seed: {SEED}", flush=True)

    preprocess_data = load_original_preprocessor()
    x, y = preprocess_data(str(DATA_PATH))
    print(f"Samples: {len(x)}; features: {x.shape[1]}; positive_rate: {y.mean():.6f}", flush=True)

    selected, tuning = tune_learning_rate(x, y, device)
    print(f"Selected parameters: {selected}", flush=True)
    folds, losses, predictions, roc_data = run_cross_validation(x, y, selected, device)

    summary = pd.DataFrame([
        {"metric": metric, "mean": folds[metric].mean(), "std": folds[metric].std(ddof=0)}
        for metric in ["accuracy", "precision", "recall", "f1", "auc"]
    ])
    params = pd.DataFrame([
        {"item": "Random seed", "original": "Not explicitly set", "adjusted": SEED, "reason": "Keep the project seed fixed and make runs reproducible"},
        {"item": "Classification output", "original": "128 LSTM hidden values", "adjusted": "Linear(128, 2)", "reason": "Correct missing binary classification head"},
        {"item": "Optimizer", "original": "Adam", "adjusted": selected["optimizer"], "reason": "Selected on inner chronological validation"},
        {"item": "Learning rate", "original": 1e-5, "adjusted": selected["lr"], "reason": "Selected on inner chronological validation; 归一.py reference is 5e-4"},
        {"item": "Weight decay", "original": 0.0, "adjusted": WEIGHT_DECAY if selected["optimizer"] == "AdamW" else 0.0, "reason": "Matches 归一.py when AdamW is selected"},
        {"item": "LR scheduler", "original": "None", "adjusted": "CosineAnnealingLR(T_max=100)", "reason": "Stable decay after selecting the initial learning rate"},
        {"item": "Gradient clipping", "original": "None", "adjusted": 1.0, "reason": "Limit recurrent-network gradient spikes"},
        {"item": "Epochs", "original": 100, "adjusted": MAX_EPOCHS, "reason": "Unchanged"},
        {"item": "Batch size", "original": 32, "adjusted": BATCH_SIZE, "reason": "Unchanged"},
        {"item": "Hidden size / LSTM layers / dropout", "original": "128 / 3 / 0.3", "adjusted": "128 / 3 / 0.3", "reason": "Unchanged"},
        {"item": "TimeSeriesSplit", "original": "5 folds", "adjusted": "5 folds", "reason": "Unchanged"},
        {"item": "Loss", "original": "CrossEntropyLoss", "adjusted": "CrossEntropyLoss", "reason": "Unchanged"},
    ])

    tuning.to_csv(OUTPUT_DIR / "tuning_results.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    losses.to_csv(OUTPUT_DIR / "epoch_loss.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    params.to_csv(OUTPUT_DIR / "parameter_changes.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUTPUT_DIR / "CNN-LSTM_results.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        folds.to_excel(writer, sheet_name="fold_metrics", index=False)
        tuning.to_excel(writer, sheet_name="tuning", index=False)
        params.to_excel(writer, sheet_name="parameters", index=False)
        losses.to_excel(writer, sheet_name="epoch_loss", index=False)
        predictions.to_excel(writer, sheet_name="predictions", index=False)
    save_plots(losses, roc_data)

    environment = {
        "seed": SEED,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "data_path": str(DATA_PATH),
        "samples": len(x),
        "features": x.shape[1],
        "selected": selected,
    }
    (OUTPUT_DIR / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FINAL SUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
