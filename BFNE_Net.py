import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, mean_squared_error,
    mean_absolute_percentage_error
)
from regression_scale_utils import regression_metrics_original_price
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor
)
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from torch.utils.data import Dataset, DataLoader
import logging
import matplotlib.pyplot as plt
import ta
import warnings
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import autocast, GradScaler
import optuna

warnings.filterwarnings('ignore')  # silence warnings

# fix seeds
torch.manual_seed(42)
np.random.seed(42)

# logging
logging.basicConfig(
    filename='training_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# regression dataset
class FinancialRegressionDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets  # already scaled in main

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )

# classification dataset
class FinancialClassificationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels.values.astype(np.int64)  # np.long is gone in newer numpy, use int64

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# early stopping
class EarlyStopping:
    def __init__(self, patience=15, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# focal loss for classification
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            alpha = self.alpha[targets].unsqueeze(1)
            focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# FCNN for regression
class FCNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout=0.5):
        super(FCNNRegressor, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# FCNN for classification
class FCNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=2, dropout=0.5):
        super(FCNNClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# RMSE
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# MAPE
def mean_absolute_percentage_error_custom(y_true, y_pred):
    """MAPE as a percentage (2.5 means 2.5%)."""
    return mean_absolute_percentage_error(y_true, y_pred) * 100.0

# preprocessing
def preprocess_data(file_path):
    # load data
    data = pd.read_excel(file_path)

    # make Date a datetime
    if not np.issubdtype(data['Date'].dtype, np.datetime64):
        data['Date'] = pd.to_datetime(data['Date'])

    # class label: 1 if price goes up
    data['Next_Day_Change'] = (data['GOLD'].diff() > 0).astype(int)

    # regression target: next day's gold price
    data['Next_Day_Price'] = data['GOLD'].shift(-1)

    # feature engineering
    # lags
    for lag in range(1, 31):
        data[f'GOLD_lag{lag}'] = data['GOLD'].shift(lag)

    # moving averages
    data['GOLD_MA_3_days'] = data['GOLD'].rolling(window=3).mean()
    data['GOLD_MA_5_days'] = data['GOLD'].rolling(window=5).mean()
    data['GOLD_MA_10_days'] = data['GOLD'].rolling(window=10).mean()

    # technical indicators
    data['RSI'] = ta.momentum.RSIIndicator(close=data['GOLD'], window=14).rsi()
    macd = ta.trend.MACD(close=data['GOLD'])
    data['MACD'] = macd.macd_diff()

    # bollinger bands
    bollinger = ta.volatility.BollingerBands(close=data['GOLD'], window=20)
    data['Bollinger_High'] = bollinger.bollinger_hband()
    data['Bollinger_Low'] = bollinger.bollinger_lband()

    # rolling stats
    rolling_window = 10
    data['GOLD_roll_mean_5'] = data['GOLD'].rolling(window=rolling_window).mean()
    data['GOLD_roll_std_5'] = data['GOLD'].rolling(window=rolling_window).std()
    data['GOLD_roll_min_5'] = data['GOLD'].rolling(window=rolling_window).min()
    data['GOLD_roll_max_5'] = data['GOLD'].rolling(window=rolling_window).max()

    # custom features
    data['Gold_Oil_Ratio'] = data['GOLD'] / data['CrudeOil_SpotPrice_BrentUK']
    data['USD_CNY_to_JPY'] = data['SpotRate_USD_CNY'] / data['SpotRate_Tokyo_9AM_USD_JPY']
    data['China_US_CPI_Ratio'] = data['China_CPI_YoY_CurrentMonth'] / data['US_CPI_YoY_NSA']

    # rate of change
    data['Gold_Rate_of_Change'] = data['GOLD'].pct_change(periods=5)
    data['Oil_Rate_of_Change'] = data['CrudeOil_SpotPrice_BrentUK'].pct_change(periods=5)

    # trend
    data['Gold_Trend_7_days'] = data['GOLD'].rolling(window=7).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )

    # inflation-adjusted gold price
    data['Gold_Inflation_Adjusted'] = data['GOLD'] / data['US_CPI_YoY_NSA']

    # volatility
    data['Gold_Volatility_10_days'] = data['GOLD'].rolling(window=10).std()
    data['Oil_Volatility_10_days'] = data['CrudeOil_SpotPrice_BrentUK'].rolling(window=10).std()

    # interactions
    data['Gold_SP500_Ratio'] = data['GOLD'] / data['US_SP500_Index']
    data['US_Japan_Interest_Rate_Diff'] = data['US_DowJones_IndustrialAverage'] - data['SpotRate_Tokyo_9AM_USD_JPY']

    # time features
    data['Month'] = data['Date'].dt.month
    data['Quarter'] = data['Date'].dt.quarter
    data['Day_of_Week'] = data['Date'].dt.dayofweek

    # shift same-day features back one day
    today_features = [
        'Gold_Oil_Ratio', 'USD_CNY_to_JPY', 'China_US_CPI_Ratio',
        'Gold_Rate_of_Change', 'Oil_Rate_of_Change', 'Gold_Trend_7_days',
        'Gold_Inflation_Adjusted', 'Gold_Volatility_10_days', 'Oil_Volatility_10_days',
        'Gold_SP500_Ratio', 'US_Japan_Interest_Rate_Diff'
    ]
    data[today_features] = data[today_features].shift(1)

    # drop NaN / inf
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    # feature list
    features = [
        'GOLD_MA_3_days', 'GOLD_lag1', 'GOLD_MA_5_days',
        'GOLD_lag2', 'GOLD_MA_10_days', 'GOLD_lag3', 'GOLD_lag4', 'GOLD_lag5',
        'GOLD_lag6', 'GOLD_lag7', 'GOLD_lag8', 'GOLD_lag9', 'GOLD_lag10',
        'GOLD_lag11', 'GOLD_lag12', 'GOLD_lag13', 'GOLD_lag14', 'GOLD_lag15',
        'Gold_Oil_Ratio', 'USD_CNY_to_JPY', 'China_US_CPI_Ratio',
        'Gold_Rate_of_Change', 'Oil_Rate_of_Change', 'Gold_Trend_7_days',
        'Gold_Inflation_Adjusted', 'Gold_Volatility_10_days', 'Oil_Volatility_10_days',
        'Gold_SP500_Ratio', 'US_Japan_Interest_Rate_Diff',
        'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low',
        'GOLD_roll_mean_5', 'GOLD_roll_std_5', 'GOLD_roll_min_5', 'GOLD_roll_max_5',
        'CrudeOil_SpotPrice_BrentUK', 'China_CPI_YoY_CurrentMonth', 'US_CPI_YoY_NSA',
        'Japan_CPI_YoY', 'SaudiArabia_CPI_YoY', 'Turkey_CPI_YoY',
        'US_DowJones_IndustrialAverage', 'US_SP500_Index', 'SpotRate_USD_CNY',
        'Interbank_OpenRate_USD_INR', 'SpotRate_Tokyo_9AM_USD_JPY'
    ]

    # make sure every feature exists
    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        raise ValueError(f"The following required features are missing from the data: {missing_features}")

    # split X and y
    X = data[features]
    y_class = data['Next_Day_Change']
    y_reg = data['Next_Day_Price']

    return X, y_class, y_reg

# scale features
def preprocess_and_scale(X_train, X_test, scaler_type='StandardScaler'):
    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_type == 'RobustScaler':
        scaler = RobustScaler()
    else:
        raise ValueError("Unsupported scaler type. Choose 'StandardScaler' or 'RobustScaler'.")

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

# scale the regression target
def preprocess_and_scale_reg_target(y_train, y_test, scaler_type='StandardScaler'):
    if scaler_type == 'StandardScaler':
        scaler_y = StandardScaler()
    elif scaler_type == 'RobustScaler':
        scaler_y = RobustScaler()
    else:
        raise ValueError("Unsupported scaler type. Choose 'StandardScaler' or 'RobustScaler'.")

    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
    return y_train_scaled, y_test_scaled, scaler_y

# optuna objective, LightGBM regression
def objective_lgbm_reg(trial, X_train, y_train):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'random_state': 42,
        'n_jobs': -1
    }

    lgbm = LGBMRegressor(**param)
    score = cross_val_score(lgbm, X_train, y_train, cv=3, scoring='neg_mean_squared_error').mean()
    return -score

# tune LightGBM (regression)
def optimize_lgbm_reg(X_train, y_train):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_lgbm_reg(trial, X_train, y_train), n_trials=50)

    print("Best parameters for LGBMRegressor:", study.best_params)
    print("Best MSE for LGBMRegressor:", study.best_value)
    logging.info(f"Best parameters for LGBMRegressor: {study.best_params}")
    logging.info(f"Best MSE for LGBMRegressor: {study.best_value}")

    return study.best_params

# optuna objective, LightGBM classification
def objective_lgbm_clf(trial, X_train, y_train):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'random_state': 42,
        'n_jobs': -1
    }

    lgbm = LGBMClassifier(**param)
    score = cross_val_score(lgbm, X_train, y_train, cv=3, scoring='roc_auc').mean()
    return score

# tune LightGBM (classification)
def optimize_lgbm_clf(X_train, y_train):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_lgbm_clf(trial, X_train, y_train), n_trials=50)

    print("Best parameters for LGBMClassifier:", study.best_params)
    print("Best ROC AUC for LGBMClassifier:", study.best_value)
    logging.info(f"Best parameters for LGBMClassifier: {study.best_params}")
    logging.info(f"Best ROC AUC for LGBMClassifier: {study.best_value}")

    return study.best_params

# base models, regression
def train_base_models_reg(X_train, y_train, device):
    hidden_dims1 = [256, 128, 64]
    fcnn1 = FCNNRegressor(input_dim=X_train.shape[1], hidden_dims=hidden_dims1).to(device)

    hidden_dims2 = [512, 256, 128, 64]
    fcnn2 = FCNNRegressor(input_dim=X_train.shape[1], hidden_dims=hidden_dims2).to(device)

    # tune LightGBM first
    best_params_lgbm = optimize_lgbm_reg(X_train, y_train)
    lgbm_reg = LGBMRegressor(**best_params_lgbm, random_state=42, n_jobs=-1)

    xgb_reg = XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    rf_reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    optimizer1 = optim.AdamW(fcnn1.parameters(), lr=0.005, weight_decay=1e-5)
    optimizer2 = optim.AdamW(fcnn2.parameters(), lr=0.005, weight_decay=1e-5)

    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=50)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=50)

    early_stopping1 = EarlyStopping(patience=12)
    early_stopping2 = EarlyStopping(patience=12)

    mse_loss = nn.MSELoss()

    models = (fcnn1, fcnn2, lgbm_reg, xgb_reg, rf_reg)
    optimizers = (optimizer1, optimizer2)
    schedulers = (scheduler1, scheduler2)
    early_stoppings = (early_stopping1, early_stopping2)

    return models, optimizers, schedulers, early_stoppings, mse_loss

# base models, classification
def train_base_models_clf(X_train, y_train, device):
    hidden_dims1 = [256, 128, 64]
    fcnn1 = FCNNClassifier(input_dim=X_train.shape[1], hidden_dims=hidden_dims1).to(device)

    hidden_dims2 = [512, 256, 128, 64]
    fcnn2 = FCNNClassifier(input_dim=X_train.shape[1], hidden_dims=hidden_dims2).to(device)

    # tune LightGBM first
    best_params_lgbm = optimize_lgbm_clf(X_train, y_train)
    lgbm_clf = LGBMClassifier(**best_params_lgbm, random_state=42, n_jobs=-1)

    xgb_clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    optimizer1 = optim.AdamW(fcnn1.parameters(), lr=0.0005, weight_decay=1e-5)
    optimizer2 = optim.AdamW(fcnn2.parameters(), lr=0.0005, weight_decay=1e-5)

    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=50)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=50)

    early_stopping1 = EarlyStopping(patience=15)
    early_stopping2 = EarlyStopping(patience=15)

    # class weights + focal loss
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    focal_loss = FocalLoss(alpha=class_weights, gamma=2)

    models = (fcnn1, fcnn2, lgbm_clf, xgb_clf, rf_clf)
    optimizers = (optimizer1, optimizer2)
    schedulers = (scheduler1, scheduler2)
    early_stoppings = (early_stopping1, early_stopping2)

    return models, optimizers, schedulers, early_stoppings, focal_loss

# train regression FCNNs
def train_fcnn_reg(models, optimizers, schedulers, early_stoppings, mse_loss, train_loader, device, num_epochs=300):
    fcnn1, fcnn2, _, _, _ = models
    optimizer1, optimizer2 = optimizers
    scheduler1, scheduler2 = schedulers
    early_stopping1, early_stopping2 = early_stoppings

    loss_history1 = []
    loss_history2 = []
    scaler = GradScaler()

    for epoch in range(num_epochs):
        fcnn1.train()
        fcnn2.train()
        running_loss1 = 0.0
        running_loss2 = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer1.zero_grad()
            with autocast():
                outputs1 = fcnn1(X_batch).squeeze()
                loss1 = mse_loss(outputs1, y_batch)
            scaler.scale(loss1).backward()
            scaler.step(optimizer1)
            scaler.update()
            running_loss1 += loss1.item()

            optimizer2.zero_grad()
            with autocast():
                outputs2 = fcnn2(X_batch).squeeze()
                loss2 = mse_loss(outputs2, y_batch)
            scaler.scale(loss2).backward()
            scaler.step(optimizer2)
            scaler.update()
            running_loss2 += loss2.item()

        avg_loss1 = running_loss1 / len(train_loader)
        avg_loss2 = running_loss2 / len(train_loader)
        loss_history1.append(avg_loss1)
        loss_history2.append(avg_loss2)

        print(f"Epoch {epoch + 1} - FCNN1 Loss: {avg_loss1:.4f}, FCNN2 Loss: {avg_loss2:.4f}")
        logging.info(f"Epoch {epoch + 1} - FCNN1 Loss: {avg_loss1:.4f}, FCNN2 Loss: {avg_loss2:.4f}")

        scheduler1.step()
        scheduler2.step()

        # early stopping
        early_stopping1(avg_loss1)
        early_stopping2(avg_loss2)
        if early_stopping1.early_stop or early_stopping2.early_stop:
            print("Early stopping triggered.")
            logging.info("Early stopping triggered.")
            break

    return loss_history1, loss_history2

# train classification FCNNs
def train_fcnn_clf(models, optimizers, schedulers, early_stoppings, focal_loss, train_loader, device, num_epochs=300):
    fcnn1, fcnn2, _, _, _ = models
    optimizer1, optimizer2 = optimizers
    scheduler1, scheduler2 = schedulers
    early_stopping1, early_stopping2 = early_stoppings

    loss_history1 = []
    loss_history2 = []
    scaler = GradScaler()

    for epoch in range(num_epochs):
        fcnn1.train()
        fcnn2.train()
        running_loss1 = 0.0
        running_loss2 = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer1.zero_grad()
            with autocast():
                outputs1 = fcnn1(X_batch)
                loss1 = focal_loss(outputs1, y_batch)
            scaler.scale(loss1).backward()
            scaler.step(optimizer1)
            scaler.update()
            running_loss1 += loss1.item()

            optimizer2.zero_grad()
            with autocast():
                outputs2 = fcnn2(X_batch)
                loss2 = focal_loss(outputs2, y_batch)
            scaler.scale(loss2).backward()
            scaler.step(optimizer2)
            scaler.update()
            running_loss2 += loss2.item()

        avg_loss1 = running_loss1 / len(train_loader)
        avg_loss2 = running_loss2 / len(train_loader)
        loss_history1.append(avg_loss1)
        loss_history2.append(avg_loss2)

        print(f"Epoch {epoch + 1} - FCNN1 Loss: {avg_loss1:.4f}, FCNN2 Loss: {avg_loss2:.4f}")
        logging.info(f"Epoch {epoch + 1} - FCNN1 Loss: {avg_loss1:.4f}, FCNN2 Loss: {avg_loss2:.4f}")

        scheduler1.step()
        scheduler2.step()

        # early stopping
        early_stopping1(avg_loss1)
        early_stopping2(avg_loss2)
        if early_stopping1.early_stop or early_stopping2.early_stop:
            print("Early stopping triggered.")
            logging.info("Early stopping triggered.")
            break

    return loss_history1, loss_history2

# loss curves
def plot_loss_curve(loss_history1, loss_history2, fold, task='regression', eval=False):
    plt.figure(figsize=(10, 6))
    if task in ['regression', 'classification']:
        plt.plot(loss_history1, label='FCNN1 Loss', color='blue', linestyle='-')
        plt.plot(loss_history2, label='FCNN2 Loss', color='orange', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    title = f'Training Loss Curve - Fold {fold + 1} - {task.capitalize()}'
    if eval:
        title += ' (Evaluation)'
    plt.title(title)
    plt.legend()
    plt.grid(True)
    filename = f'training_loss_curve_fold_{fold + 1}_{task}.png'
    if eval:
        filename = f'training_loss_curve_eval_fold_{fold + 1}_{task}.png'
    plt.savefig(filename)
    plt.close()

# meta-features, regression
def generate_meta_features_reg(fcnn1, fcnn2, lgbm_reg, xgb_reg, rf_reg, test_loader, X_test_scaled, device):
    fcnn1.eval()
    fcnn2.eval()
    preds1, preds2 = [], []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            output1 = fcnn1(X_batch).squeeze().cpu().numpy()
            output2 = fcnn2(X_batch).squeeze().cpu().numpy()
            preds1.extend(output1)
            preds2.extend(output2)

    lgbm_preds = lgbm_reg.predict(X_test_scaled)

    xgb_preds = xgb_reg.predict(X_test_scaled)

    rf_preds = rf_reg.predict(X_test_scaled)

    # stack into meta-features
    fold_meta_features = np.column_stack([
        preds1,
        preds2,
        lgbm_preds,
        xgb_preds,
        rf_preds
    ])
    return fold_meta_features

# meta-features, classification
def generate_meta_features_clf(fcnn1, fcnn2, lgbm_clf, xgb_clf, rf_clf, test_loader, X_test_scaled, device):
    fcnn1.eval()
    fcnn2.eval()
    preds1, preds2 = [], []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            output1 = fcnn1(X_batch)
            output2 = fcnn2(X_batch)
            preds1.extend(torch.softmax(output1, dim=1).cpu().numpy())
            preds2.extend(torch.softmax(output2, dim=1).cpu().numpy())

    lgbm_probs = lgbm_clf.predict_proba(X_test_scaled)

    xgb_probs = xgb_clf.predict_proba(X_test_scaled)

    rf_probs = rf_clf.predict_proba(X_test_scaled)

    # stack into meta-features
    fold_meta_features = np.hstack([
        np.array(preds1),
        np.array(preds2),
        lgbm_probs,
        xgb_probs,
        rf_probs
    ])
    return fold_meta_features

def main():
    # data path, change when moving machines
    file_path = os.path.join(os.path.dirname(__file__), "GOLD_cleaned.xlsx")
    try:
        X, y_class, y_reg = preprocess_data(file_path)
    except Exception as e:
        print(f"Error in preprocessing data: {e}")
        logging.error(f"Error in preprocessing data: {e}")
        return

    # label balance
    print("Classification Target Distribution:")
    print(y_class.value_counts())
    logging.info(f"Classification Target Distribution:\n{y_class.value_counts()}")

    print("\nRegression Target Statistics:")
    print(y_reg.describe())
    logging.info(f"Regression Target Statistics:\n{y_reg.describe()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    logging.info(f"Using device: {device}")

    tscv = TimeSeriesSplit(n_splits=5)

    # first CV: collect meta-features and labels
    meta_train_features_reg_cv1 = []
    meta_train_labels_reg_cv1 = []
    meta_train_features_clf_cv1 = []
    meta_train_labels_clf_cv1 = []

    # second CV: collect eval metrics
    rmses, mapes = [], []
    accuracies, precisions, recalls, f1_scores, aucs = [], [], [], [], []

    # first CV: build the meta-features
    print("\nGenerating meta-features and training base models for Regression and Classification...")
    logging.info("Generating meta-features and training base models started.")

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\nFold {fold + 1} - Training base models")
        logging.info(f"Fold {fold + 1} - Training base models started")

        X_train_reg, X_test_reg = X.iloc[train_index], X.iloc[test_index]
        y_train_reg_fold, y_test_reg_fold = y_reg.iloc[train_index], y_reg.iloc[test_index]

        X_train_clf, X_test_clf = X.iloc[train_index], X.iloc[test_index]
        y_train_clf_fold, y_test_clf_fold = y_class.iloc[train_index], y_class.iloc[test_index]

        X_train_reg_scaled, X_test_reg_scaled, scaler_X_reg = preprocess_and_scale(
            X_train_reg, X_test_reg, scaler_type='StandardScaler'
        )

        y_train_reg_scaled, y_test_reg_scaled, scaler_y_reg = preprocess_and_scale_reg_target(
            y_train_reg_fold, y_test_reg_fold, scaler_type='StandardScaler'
        )

        X_train_clf_scaled, X_test_clf_scaled, scaler_X_clf = preprocess_and_scale(
            X_train_clf, X_test_clf, scaler_type='StandardScaler'
        )

        train_dataset_reg = FinancialRegressionDataset(X_train_reg_scaled, y_train_reg_scaled)
        test_dataset_reg = FinancialRegressionDataset(X_test_reg_scaled, y_test_reg_scaled)
        train_loader_reg = DataLoader(train_dataset_reg, batch_size=32, shuffle=True)
        test_loader_reg = DataLoader(test_dataset_reg, batch_size=32, shuffle=False)

        train_dataset_clf = FinancialClassificationDataset(X_train_clf_scaled, y_train_clf_fold)
        test_dataset_clf = FinancialClassificationDataset(X_test_clf_scaled, y_test_clf_fold)
        train_loader_clf = DataLoader(train_dataset_clf, batch_size=32, shuffle=True)
        test_loader_clf = DataLoader(test_dataset_clf, batch_size=32, shuffle=False)

        # base models, regression
        models_reg, optimizers_reg, schedulers_reg, early_stoppings_reg, mse_loss = train_base_models_reg(
            X_train_reg_scaled, y_train_reg_scaled, device
        )
        loss_history1_reg, loss_history2_reg = train_fcnn_reg(
            models=models_reg,
            optimizers=optimizers_reg,
            schedulers=schedulers_reg,
            early_stoppings=early_stoppings_reg,
            mse_loss=mse_loss,
            train_loader=train_loader_reg,
            device=device
        )

        plot_loss_curve(loss_history1_reg, loss_history2_reg, fold, task='regression')

        fcnn1_reg, fcnn2_reg, lgbm_reg, xgb_reg, rf_reg = models_reg

        try:
            lgbm_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during LGBMRegressor training: {e}")
            logging.error(f"Error during LGBMRegressor training: {e}")
            continue  # skip this fold if it errors

        try:
            xgb_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during XGBoost Regressor training: {e}")
            logging.error(f"Error during XGBoost Regressor training: {e}")
            continue

        try:
            rf_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during RandomForestRegressor training: {e}")
            logging.error(f"Error during RandomForestRegressor training: {e}")
            continue

        # meta-features, regression
        fold_meta_features_reg = generate_meta_features_reg(
            fcnn1=fcnn1_reg,
            fcnn2=fcnn2_reg,
            lgbm_reg=lgbm_reg,
            xgb_reg=xgb_reg,
            rf_reg=rf_reg,
            test_loader=test_loader_reg,
            X_test_scaled=X_test_reg_scaled,
            device=device
        )

        meta_train_features_reg_cv1.append(fold_meta_features_reg)
        meta_train_labels_reg_cv1.extend(y_test_reg_scaled)  # meta model trains on the scaled target

        # base models, classification
        models_clf, optimizers_clf, schedulers_clf, early_stoppings_clf, focal_loss = train_base_models_clf(
            X_train_clf_scaled, y_train_clf_fold, device
        )
        loss_history1_clf, loss_history2_clf = train_fcnn_clf(
            models=models_clf,
            optimizers=optimizers_clf,
            schedulers=schedulers_clf,
            early_stoppings=early_stoppings_clf,
            focal_loss=focal_loss,
            train_loader=train_loader_clf,
            device=device
        )

        plot_loss_curve(loss_history1_clf, loss_history2_clf, fold, task='classification')

        fcnn1_clf, fcnn2_clf, lgbm_clf, xgb_clf, rf_clf = models_clf

        try:
            lgbm_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during LGBMClassifier training: {e}")
            logging.error(f"Error during LGBMClassifier training: {e}")
            continue

        try:
            xgb_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during XGBoost Classifier training: {e}")
            logging.error(f"Error during XGBoost Classifier training: {e}")
            continue

        try:
            rf_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during RandomForestClassifier training: {e}")
            logging.error(f"Error during RandomForestClassifier training: {e}")
            continue

        # meta-features, classification
        fold_meta_features_clf = generate_meta_features_clf(
            fcnn1=fcnn1_clf,
            fcnn2=fcnn2_clf,
            lgbm_clf=lgbm_clf,
            xgb_clf=xgb_clf,
            rf_clf=rf_clf,
            test_loader=test_loader_clf,
            X_test_scaled=X_test_clf_scaled,
            device=device
        )

        meta_train_features_clf_cv1.append(fold_meta_features_clf)
        meta_train_labels_clf_cv1.extend(y_test_clf_fold.values)

    # combine regression meta-features
    meta_train_features_reg_cv1 = np.vstack(meta_train_features_reg_cv1)
    meta_train_labels_reg_cv1 = np.array(meta_train_labels_reg_cv1)

    # combine classification meta-features
    meta_train_features_clf_cv1 = np.vstack(meta_train_features_clf_cv1)
    meta_train_labels_clf_cv1 = np.array(meta_train_labels_clf_cv1)

    # regression meta model
    print("\nTraining Meta-Model for Regression...")
    logging.info("Training Meta-Model for Regression started.")
    meta_model_reg = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.15,
        max_depth=5,
        random_state=42
    )
    meta_model_reg.fit(meta_train_features_reg_cv1, meta_train_labels_reg_cv1)

    # classification meta model
    print("\nTraining Meta-Model for Classification...")
    logging.info("Training Meta-Model for Classification started.")
    meta_model_clf = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.15,
        max_depth=5,
        random_state=42
    )
    meta_model_clf.fit(meta_train_features_clf_cv1, meta_train_labels_clf_cv1)

    # second CV: evaluate the ensemble
    print("\nEvaluating Ensemble Models for Regression and Classification...")
    logging.info("Evaluating Ensemble Models started.")

    eval_rmses, eval_mapes = [], []
    eval_accuracies, eval_precisions, eval_recalls, eval_f1_scores, eval_aucs = [], [], [], [], []

    meta_train_features_reg_cv2 = []
    meta_train_labels_reg_cv2 = []
    meta_train_features_clf_cv2 = []
    meta_train_labels_clf_cv2 = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\nFold {fold + 1} - Evaluating Ensemble Models")
        logging.info(f"Fold {fold + 1} - Evaluating Ensemble Models started")

        X_train_reg, X_test_reg = X.iloc[train_index], X.iloc[test_index]
        y_train_reg_fold, y_test_reg_fold = y_reg.iloc[train_index], y_reg.iloc[test_index]

        X_train_clf, X_test_clf = X.iloc[train_index], X.iloc[test_index]
        y_train_clf_fold, y_test_clf_fold = y_class.iloc[train_index], y_class.iloc[test_index]

        X_train_reg_scaled, X_test_reg_scaled, scaler_X_reg_eval = preprocess_and_scale(
            X_train_reg, X_test_reg, scaler_type='StandardScaler'
        )
        y_train_reg_scaled, y_test_reg_scaled, scaler_y_reg_eval = preprocess_and_scale_reg_target(
            y_train_reg_fold, y_test_reg_fold, scaler_type='StandardScaler'
        )
        X_train_clf_scaled, X_test_clf_scaled, scaler_X_clf_eval = preprocess_and_scale(
            X_train_clf, X_test_clf, scaler_type='StandardScaler'
        )

        train_dataset_reg = FinancialRegressionDataset(X_train_reg_scaled, y_train_reg_scaled)
        test_dataset_reg = FinancialRegressionDataset(X_test_reg_scaled, y_test_reg_scaled)
        train_loader_reg = DataLoader(train_dataset_reg, batch_size=32, shuffle=True)
        test_loader_reg = DataLoader(test_dataset_reg, batch_size=32, shuffle=False)

        train_dataset_clf = FinancialClassificationDataset(X_train_clf_scaled, y_train_clf_fold)
        test_dataset_clf = FinancialClassificationDataset(X_test_clf_scaled, y_test_clf_fold)
        train_loader_clf = DataLoader(train_dataset_clf, batch_size=32, shuffle=True)
        test_loader_clf = DataLoader(test_dataset_clf, batch_size=32, shuffle=False)

        # base models, regression
        models_reg, optimizers_reg, schedulers_reg, early_stoppings_reg, mse_loss = train_base_models_reg(
            X_train_reg_scaled, y_train_reg_scaled, device
        )
        loss_history1_reg, loss_history2_reg = train_fcnn_reg(
            models=models_reg,
            optimizers=optimizers_reg,
            schedulers=schedulers_reg,
            early_stoppings=early_stoppings_reg,
            mse_loss=mse_loss,
            train_loader=train_loader_reg,
            device=device
        )

        plot_loss_curve(loss_history1_reg, loss_history2_reg, fold, task='regression', eval=True)

        fcnn1_reg, fcnn2_reg, lgbm_reg, xgb_reg, rf_reg = models_reg

        try:
            lgbm_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during LGBMRegressor training: {e}")
            logging.error(f"Error during LGBMRegressor training: {e}")
            continue

        try:
            xgb_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during XGBoost Regressor training: {e}")
            logging.error(f"Error during XGBoost Regressor training: {e}")
            continue

        try:
            rf_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during RandomForestRegressor training: {e}")
            logging.error(f"Error during RandomForestRegressor training: {e}")
            continue

        # meta-features, regression
        fold_meta_features_reg = generate_meta_features_reg(
            fcnn1=fcnn1_reg,
            fcnn2=fcnn2_reg,
            lgbm_reg=lgbm_reg,
            xgb_reg=xgb_reg,
            rf_reg=rf_reg,
            test_loader=test_loader_reg,
            X_test_scaled=X_test_reg_scaled,
            device=device
        )

        fold_meta_preds_reg_scaled = meta_model_reg.predict(fold_meta_features_reg)
        # the meta model predicts this fold's standardized target;
        # undo it once with the scaler fit on this fold's training set, then compare to raw prices.
        fold_meta_preds_reg, rmse, mape = regression_metrics_original_price(
            y_test_reg_fold,
            fold_meta_preds_reg_scaled,
            prediction_scaler=scaler_y_reg_eval,
        )

        rmses.append(rmse)
        mapes.append(mape)

        print(f"Fold {fold + 1} - Regression: RMSE (USD/oz)={rmse:.4f}, MAPE (%)={mape:.4f}")
        logging.info(f"Fold {fold + 1} - Regression: RMSE (USD/oz)={rmse:.4f}, MAPE (%)={mape:.4f}")

        # base models, classification
        models_clf, optimizers_clf, schedulers_clf, early_stoppings_clf, focal_loss = train_base_models_clf(
            X_train_clf_scaled, y_train_clf_fold, device
        )
        loss_history1_clf, loss_history2_clf = train_fcnn_clf(
            models=models_clf,
            optimizers=optimizers_clf,
            schedulers=schedulers_clf,
            early_stoppings=early_stoppings_clf,
            focal_loss=focal_loss,
            train_loader=train_loader_clf,
            device=device
        )

        plot_loss_curve(loss_history1_clf, loss_history2_clf, fold, task='classification', eval=True)

        fcnn1_clf, fcnn2_clf, lgbm_clf, xgb_clf, rf_clf = models_clf

        try:
            lgbm_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during LGBMClassifier training: {e}")
            logging.error(f"Error during LGBMClassifier training: {e}")
            continue

        try:
            xgb_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during XGBoost Classifier training: {e}")
            logging.error(f"Error during XGBoost Classifier training: {e}")
            continue

        try:
            rf_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during RandomForestClassifier training: {e}")
            logging.error(f"Error during RandomForestClassifier training: {e}")
            continue

        # meta-features, classification
        fold_meta_features_clf = generate_meta_features_clf(
            fcnn1=fcnn1_clf,
            fcnn2=fcnn2_clf,
            lgbm_clf=lgbm_clf,
            xgb_clf=xgb_clf,
            rf_clf=rf_clf,
            test_loader=test_loader_clf,
            X_test_scaled=X_test_clf_scaled,
            device=device
        )

        meta_train_features_clf_cv2.append(fold_meta_features_clf)
        meta_train_labels_clf_cv2.extend(y_test_clf_fold.values)

        # same as before: classification is only scored in the first evaluation round,
        # regression is scored in all five.
        fold_meta_preds_clf = meta_model_clf.predict(fold_meta_features_clf)
        fold_meta_probs_clf = meta_model_clf.predict_proba(fold_meta_features_clf)[:, 1]
        accuracy = accuracy_score(y_test_clf_fold, fold_meta_preds_clf)
        precision = precision_score(y_test_clf_fold, fold_meta_preds_clf, zero_division=0)
        recall = recall_score(y_test_clf_fold, fold_meta_preds_clf, zero_division=0)
        f1 = f1_score(y_test_clf_fold, fold_meta_preds_clf, zero_division=0)
        auc = roc_auc_score(y_test_clf_fold, fold_meta_probs_clf)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        aucs.append(auc)
        print(f"Fold {fold + 1} - Classification: ACC={accuracy:.4f}, PREC={precision:.4f}, "
              f"REC={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
        logging.info(f"Fold {fold + 1} - Classification: ACC={accuracy:.4f}, PREC={precision:.4f}, "
                     f"REC={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    # combine regression meta-features (first CV)
    meta_train_features_reg_cv1 = np.vstack(meta_train_features_reg_cv1)
    meta_train_labels_reg_cv1 = np.array(meta_train_labels_reg_cv1)

    # combine classification meta-features (first CV)
    meta_train_features_clf_cv1 = np.vstack(meta_train_features_clf_cv1)
    meta_train_labels_clf_cv1 = np.array(meta_train_labels_clf_cv1)

    # regression meta model
    print("\nTraining Meta-Model for Regression...")
    logging.info("Training Meta-Model for Regression started.")
    meta_model_reg = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.15,
        max_depth=5,
        random_state=42
    )
    meta_model_reg.fit(meta_train_features_reg_cv1, meta_train_labels_reg_cv1)

    # classification meta model
    print("\nTraining Meta-Model for Classification...")
    logging.info("Training Meta-Model for Classification started.")
    meta_model_clf = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.15,
        max_depth=5,
        random_state=42
    )
    meta_model_clf.fit(meta_train_features_clf_cv1, meta_train_labels_clf_cv1)

    # second CV: evaluate the ensemble
    print("\nEvaluating Ensemble Models for Regression and Classification...")
    logging.info("Evaluating Ensemble Models started.")

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\nFold {fold + 1} - Evaluating Ensemble Models")
        logging.info(f"Fold {fold + 1} - Evaluating Ensemble Models started")

        X_train_reg, X_test_reg = X.iloc[train_index], X.iloc[test_index]
        y_train_reg_fold, y_test_reg_fold = y_reg.iloc[train_index], y_reg.iloc[test_index]

        X_train_clf, X_test_clf = X.iloc[train_index], X.iloc[test_index]
        y_train_clf_fold, y_test_clf_fold = y_class.iloc[train_index], y_class.iloc[test_index]

        X_train_reg_scaled, X_test_reg_scaled, scaler_X_reg_eval = preprocess_and_scale(
            X_train_reg, X_test_reg, scaler_type='StandardScaler'
        )
        y_train_reg_scaled, y_test_reg_scaled, scaler_y_reg_eval = preprocess_and_scale_reg_target(
            y_train_reg_fold, y_test_reg_fold, scaler_type='StandardScaler'
        )
        X_train_clf_scaled, X_test_clf_scaled, scaler_X_clf_eval = preprocess_and_scale(
            X_train_clf, X_test_clf, scaler_type='StandardScaler'
        )

        train_dataset_reg = FinancialRegressionDataset(X_train_reg_scaled, y_train_reg_scaled)
        test_dataset_reg = FinancialRegressionDataset(X_test_reg_scaled, y_test_reg_scaled)
        train_loader_reg = DataLoader(train_dataset_reg, batch_size=32, shuffle=True)
        test_loader_reg = DataLoader(test_dataset_reg, batch_size=32, shuffle=False)

        train_dataset_clf = FinancialClassificationDataset(X_train_clf_scaled, y_train_clf_fold)
        test_dataset_clf = FinancialClassificationDataset(X_test_clf_scaled, y_test_clf_fold)
        train_loader_clf = DataLoader(train_dataset_clf, batch_size=32, shuffle=True)
        test_loader_clf = DataLoader(test_dataset_clf, batch_size=32, shuffle=False)

        # base models, regression
        models_reg, optimizers_reg, schedulers_reg, early_stoppings_reg, mse_loss = train_base_models_reg(
            X_train_reg_scaled, y_train_reg_scaled, device
        )
        loss_history1_reg, loss_history2_reg = train_fcnn_reg(
            models=models_reg,
            optimizers=optimizers_reg,
            schedulers=schedulers_reg,
            early_stoppings=early_stoppings_reg,
            mse_loss=mse_loss,
            train_loader=train_loader_reg,
            device=device
        )

        plot_loss_curve(loss_history1_reg, loss_history2_reg, fold, task='regression', eval=True)

        fcnn1_reg, fcnn2_reg, lgbm_reg, xgb_reg, rf_reg = models_reg

        try:
            lgbm_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during LGBMRegressor training: {e}")
            logging.error(f"Error during LGBMRegressor training: {e}")
            continue

        try:
            xgb_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during XGBoost Regressor training: {e}")
            logging.error(f"Error during XGBoost Regressor training: {e}")
            continue

        try:
            rf_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during RandomForestRegressor training: {e}")
            logging.error(f"Error during RandomForestRegressor training: {e}")
            continue

        # meta-features, regression
        fold_meta_features_reg = generate_meta_features_reg(
            fcnn1=fcnn1_reg,
            fcnn2=fcnn2_reg,
            lgbm_reg=lgbm_reg,
            xgb_reg=xgb_reg,
            rf_reg=rf_reg,
            test_loader=test_loader_reg,
            X_test_scaled=X_test_reg_scaled,
            device=device
        )

        fold_meta_preds_reg_scaled = meta_model_reg.predict(fold_meta_features_reg)
        # back to raw prices for this fold
        fold_meta_preds_reg, rmse, mape = regression_metrics_original_price(
            y_test_reg_fold,
            fold_meta_preds_reg_scaled,
            prediction_scaler=scaler_y_reg_eval,
        )

        rmses.append(rmse)
        mapes.append(mape)

        print(f"Fold {fold + 1} - Regression: RMSE (USD/oz)={rmse:.4f}, MAPE (%)={mape:.4f}")
        logging.info(f"Fold {fold + 1} - Regression: RMSE (USD/oz)={rmse:.4f}, MAPE (%)={mape:.4f}")

        # base models, classification
        models_clf, optimizers_clf, schedulers_clf, early_stoppings_clf, focal_loss = train_base_models_clf(
            X_train_clf_scaled, y_train_clf_fold, device
        )
        loss_history1_clf, loss_history2_clf = train_fcnn_clf(
            models=models_clf,
            optimizers=optimizers_clf,
            schedulers=schedulers_clf,
            early_stoppings=early_stoppings_clf,
            focal_loss=focal_loss,
            train_loader=train_loader_clf,
            device=device
        )

        plot_loss_curve(loss_history1_clf, loss_history2_clf, fold, task='classification', eval=True)

        fcnn1_clf, fcnn2_clf, lgbm_clf, xgb_clf, rf_clf = models_clf

        try:
            lgbm_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during LGBMClassifier training: {e}")
            logging.error(f"Error during LGBMClassifier training: {e}")
            continue

        try:
            xgb_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during XGBoost Classifier training: {e}")
            logging.error(f"Error during XGBoost Classifier training: {e}")
            continue

        try:
            rf_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during RandomForestClassifier training: {e}")
            logging.error(f"Error during RandomForestClassifier training: {e}")
            continue

        # meta-features, classification
        fold_meta_features_clf = generate_meta_features_clf(
            fcnn1=fcnn1_clf,
            fcnn2=fcnn2_clf,
            lgbm_clf=lgbm_clf,
            xgb_clf=xgb_clf,
            rf_clf=rf_clf,
            test_loader=test_loader_clf,
            X_test_scaled=X_test_clf_scaled,
            device=device
        )

        meta_train_features_clf_cv2.append(fold_meta_features_clf)
        meta_train_labels_clf_cv2.extend(y_test_clf_fold.values)

    # combine regression meta-features (first CV)
    meta_train_features_reg_cv1 = np.vstack(meta_train_features_reg_cv1)
    meta_train_labels_reg_cv1 = np.array(meta_train_labels_reg_cv1)

    # combine classification meta-features (first CV)
    meta_train_features_clf_cv1 = np.vstack(meta_train_features_clf_cv1)
    meta_train_labels_clf_cv1 = np.array(meta_train_labels_clf_cv1)

    # regression meta model
    print("\nTraining Meta-Model for Regression...")
    logging.info("Training Meta-Model for Regression started.")
    meta_model_reg = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.15,
        max_depth=5,
        random_state=42
    )
    meta_model_reg.fit(meta_train_features_reg_cv1, meta_train_labels_reg_cv1)

    # classification meta model
    print("\nTraining Meta-Model for Classification...")
    logging.info("Training Meta-Model for Classification started.")
    meta_model_clf = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.15,
        max_depth=5,
        random_state=42
    )
    meta_model_clf.fit(meta_train_features_clf_cv1, meta_train_labels_clf_cv1)

    # second CV: evaluate the ensemble
    print("\nEvaluating Ensemble Models for Regression and Classification...")
    logging.info("Evaluating Ensemble Models started.")

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\nFold {fold + 1} - Evaluating Ensemble Models")
        logging.info(f"Fold {fold + 1} - Evaluating Ensemble Models started")

        X_train_reg, X_test_reg = X.iloc[train_index], X.iloc[test_index]
        y_train_reg_fold, y_test_reg_fold = y_reg.iloc[train_index], y_reg.iloc[test_index]

        X_train_clf, X_test_clf = X.iloc[train_index], X.iloc[test_index]
        y_train_clf_fold, y_test_clf_fold = y_class.iloc[train_index], y_class.iloc[test_index]

        X_train_reg_scaled, X_test_reg_scaled, scaler_X_reg_eval = preprocess_and_scale(
            X_train_reg, X_test_reg, scaler_type='StandardScaler'
        )
        y_train_reg_scaled, y_test_reg_scaled, scaler_y_reg_eval = preprocess_and_scale_reg_target(
            y_train_reg_fold, y_test_reg_fold, scaler_type='StandardScaler'
        )
        X_train_clf_scaled, X_test_clf_scaled, scaler_X_clf_eval = preprocess_and_scale(
            X_train_clf, X_test_clf, scaler_type='StandardScaler'
        )

        train_dataset_reg = FinancialRegressionDataset(X_train_reg_scaled, y_train_reg_scaled)
        test_dataset_reg = FinancialRegressionDataset(X_test_reg_scaled, y_test_reg_scaled)
        train_loader_reg = DataLoader(train_dataset_reg, batch_size=32, shuffle=True)
        test_loader_reg = DataLoader(test_dataset_reg, batch_size=32, shuffle=False)

        train_dataset_clf = FinancialClassificationDataset(X_train_clf_scaled, y_train_clf_fold)
        test_dataset_clf = FinancialClassificationDataset(X_test_clf_scaled, y_test_clf_fold)
        train_loader_clf = DataLoader(train_dataset_clf, batch_size=32, shuffle=True)
        test_loader_clf = DataLoader(test_dataset_clf, batch_size=32, shuffle=False)

        # base models, regression
        models_reg, optimizers_reg, schedulers_reg, early_stoppings_reg, mse_loss = train_base_models_reg(
            X_train_reg_scaled, y_train_reg_scaled, device
        )
        loss_history1_reg, loss_history2_reg = train_fcnn_reg(
            models=models_reg,
            optimizers=optimizers_reg,
            schedulers=schedulers_reg,
            early_stoppings=early_stoppings_reg,
            mse_loss=mse_loss,
            train_loader=train_loader_reg,
            device=device
        )

        plot_loss_curve(loss_history1_reg, loss_history2_reg, fold, task='regression', eval=True)

        fcnn1_reg, fcnn2_reg, lgbm_reg, xgb_reg, rf_reg = models_reg

        try:
            lgbm_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during LGBMRegressor training: {e}")
            logging.error(f"Error during LGBMRegressor training: {e}")
            continue

        try:
            xgb_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during XGBoost Regressor training: {e}")
            logging.error(f"Error during XGBoost Regressor training: {e}")
            continue

        try:
            rf_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during RandomForestRegressor training: {e}")
            logging.error(f"Error during RandomForestRegressor training: {e}")
            continue

        # meta-features, regression
        fold_meta_features_reg = generate_meta_features_reg(
            fcnn1=fcnn1_reg,
            fcnn2=fcnn2_reg,
            lgbm_reg=lgbm_reg,
            xgb_reg=xgb_reg,
            rf_reg=rf_reg,
            test_loader=test_loader_reg,
            X_test_scaled=X_test_reg_scaled,
            device=device
        )

        fold_meta_preds_reg_scaled = meta_model_reg.predict(fold_meta_features_reg)
        # back to raw prices for this fold
        fold_meta_preds_reg, rmse, mape = regression_metrics_original_price(
            y_test_reg_fold,
            fold_meta_preds_reg_scaled,
            prediction_scaler=scaler_y_reg_eval,
        )

        rmses.append(rmse)
        mapes.append(mape)

        print(f"Fold {fold + 1} - Regression: RMSE (USD/oz)={rmse:.4f}, MAPE (%)={mape:.4f}")
        logging.info(f"Fold {fold + 1} - Regression: RMSE (USD/oz)={rmse:.4f}, MAPE (%)={mape:.4f}")

        # base models, classification
        models_clf, optimizers_clf, schedulers_clf, early_stoppings_clf, focal_loss = train_base_models_clf(
            X_train_clf_scaled, y_train_clf_fold, device
        )
        loss_history1_clf, loss_history2_clf = train_fcnn_clf(
            models=models_clf,
            optimizers=optimizers_clf,
            schedulers=schedulers_clf,
            early_stoppings=early_stoppings_clf,
            focal_loss=focal_loss,
            train_loader=train_loader_clf,
            device=device
        )

        plot_loss_curve(loss_history1_clf, loss_history2_clf, fold, task='classification', eval=True)

        fcnn1_clf, fcnn2_clf, lgbm_clf, xgb_clf, rf_clf = models_clf

        try:
            lgbm_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during LGBMClassifier training: {e}")
            logging.error(f"Error during LGBMClassifier training: {e}")
            continue

        try:
            xgb_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during XGBoost Classifier training: {e}")
            logging.error(f"Error during XGBoost Classifier training: {e}")
            continue

        try:
            rf_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during RandomForestClassifier training: {e}")
            logging.error(f"Error during RandomForestClassifier training: {e}")
            continue

        # meta-features, classification
        fold_meta_features_clf = generate_meta_features_clf(
            fcnn1=fcnn1_clf,
            fcnn2=fcnn2_clf,
            lgbm_clf=lgbm_clf,
            xgb_clf=xgb_clf,
            rf_clf=rf_clf,
            test_loader=test_loader_clf,
            X_test_scaled=X_test_clf_scaled,
            device=device
        )

        meta_train_features_clf_cv2.append(fold_meta_features_clf)
        meta_train_labels_clf_cv2.extend(y_test_clf_fold.values)

    # combine regression meta-features (first CV)
    meta_train_features_reg_cv1 = np.vstack(meta_train_features_reg_cv1)
    meta_train_labels_reg_cv1 = np.array(meta_train_labels_reg_cv1)

    # combine classification meta-features (first CV)
    meta_train_features_clf_cv1 = np.vstack(meta_train_features_clf_cv1)
    meta_train_labels_clf_cv1 = np.array(meta_train_labels_clf_cv1)

    # regression meta model
    print("\nTraining Meta-Model for Regression...")
    logging.info("Training Meta-Model for Regression started.")
    meta_model_reg = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.15,
        max_depth=5,
        random_state=42
    )
    meta_model_reg.fit(meta_train_features_reg_cv1, meta_train_labels_reg_cv1)

    # classification meta model
    print("\nTraining Meta-Model for Classification...")
    logging.info("Training Meta-Model for Classification started.")
    meta_model_clf = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.15,
        max_depth=5,
        random_state=42
    )
    meta_model_clf.fit(meta_train_features_clf_cv1, meta_train_labels_clf_cv1)

    # second CV: evaluate the ensemble
    print("\nEvaluating Ensemble Models for Regression and Classification...")
    logging.info("Evaluating Ensemble Models started.")

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\nFold {fold + 1} - Evaluating Ensemble Models")
        logging.info(f"Fold {fold + 1} - Evaluating Ensemble Models started")

        X_train_reg, X_test_reg = X.iloc[train_index], X.iloc[test_index]
        y_train_reg_fold, y_test_reg_fold = y_reg.iloc[train_index], y_reg.iloc[test_index]

        X_train_clf, X_test_clf = X.iloc[train_index], X.iloc[test_index]
        y_train_clf_fold, y_test_clf_fold = y_class.iloc[train_index], y_class.iloc[test_index]

        X_train_reg_scaled, X_test_reg_scaled, scaler_X_reg_eval = preprocess_and_scale(
            X_train_reg, X_test_reg, scaler_type='StandardScaler'
        )
        y_train_reg_scaled, y_test_reg_scaled, scaler_y_reg_eval = preprocess_and_scale_reg_target(
            y_train_reg_fold, y_test_reg_fold, scaler_type='StandardScaler'
        )
        X_train_clf_scaled, X_test_clf_scaled, scaler_X_clf_eval = preprocess_and_scale(
            X_train_clf, X_test_clf, scaler_type='StandardScaler'
        )

        train_dataset_reg = FinancialRegressionDataset(X_train_reg_scaled, y_train_reg_scaled)
        test_dataset_reg = FinancialRegressionDataset(X_test_reg_scaled, y_test_reg_scaled)
        train_loader_reg = DataLoader(train_dataset_reg, batch_size=32, shuffle=True)
        test_loader_reg = DataLoader(test_dataset_reg, batch_size=32, shuffle=False)

        train_dataset_clf = FinancialClassificationDataset(X_train_clf_scaled, y_train_clf_fold)
        test_dataset_clf = FinancialClassificationDataset(X_test_clf_scaled, y_test_clf_fold)
        train_loader_clf = DataLoader(train_dataset_clf, batch_size=32, shuffle=True)
        test_loader_clf = DataLoader(test_dataset_clf, batch_size=32, shuffle=False)

        # base models, regression
        models_reg, optimizers_reg, schedulers_reg, early_stoppings_reg, mse_loss = train_base_models_reg(
            X_train_reg_scaled, y_train_reg_scaled, device
        )
        loss_history1_reg, loss_history2_reg = train_fcnn_reg(
            models=models_reg,
            optimizers=optimizers_reg,
            schedulers=schedulers_reg,
            early_stoppings=early_stoppings_reg,
            mse_loss=mse_loss,
            train_loader=train_loader_reg,
            device=device
        )

        plot_loss_curve(loss_history1_reg, loss_history2_reg, fold, task='regression', eval=True)

        fcnn1_reg, fcnn2_reg, lgbm_reg, xgb_reg, rf_reg = models_reg

        try:
            lgbm_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during LGBMRegressor training: {e}")
            logging.error(f"Error during LGBMRegressor training: {e}")
            continue

        try:
            xgb_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during XGBoost Regressor training: {e}")
            logging.error(f"Error during XGBoost Regressor training: {e}")
            continue

        try:
            rf_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during RandomForestRegressor training: {e}")
            logging.error(f"Error during RandomForestRegressor training: {e}")
            continue

        # meta-features, regression
        fold_meta_features_reg = generate_meta_features_reg(
            fcnn1=fcnn1_reg,
            fcnn2=fcnn2_reg,
            lgbm_reg=lgbm_reg,
            xgb_reg=xgb_reg,
            rf_reg=rf_reg,
            test_loader=test_loader_reg,
            X_test_scaled=X_test_reg_scaled,
            device=device
        )

        fold_meta_preds_reg_scaled = meta_model_reg.predict(fold_meta_features_reg)
        # back to raw prices for this fold
        fold_meta_preds_reg, rmse, mape = regression_metrics_original_price(
            y_test_reg_fold,
            fold_meta_preds_reg_scaled,
            prediction_scaler=scaler_y_reg_eval,
        )

        rmses.append(rmse)
        mapes.append(mape)

        print(f"Fold {fold + 1} - Regression: RMSE (USD/oz)={rmse:.4f}, MAPE (%)={mape:.4f}")
        logging.info(f"Fold {fold + 1} - Regression: RMSE (USD/oz)={rmse:.4f}, MAPE (%)={mape:.4f}")

        # base models, classification
        models_clf, optimizers_clf, schedulers_clf, early_stoppings_clf, focal_loss = train_base_models_clf(
            X_train_clf_scaled, y_train_clf_fold, device
        )
        loss_history1_clf, loss_history2_clf = train_fcnn_clf(
            models=models_clf,
            optimizers=optimizers_clf,
            schedulers=schedulers_clf,
            early_stoppings=early_stoppings_clf,
            focal_loss=focal_loss,
            train_loader=train_loader_clf,
            device=device
        )

        plot_loss_curve(loss_history1_clf, loss_history2_clf, fold, task='classification', eval=True)

        fcnn1_clf, fcnn2_clf, lgbm_clf, xgb_clf, rf_clf = models_clf

        try:
            lgbm_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during LGBMClassifier training: {e}")
            logging.error(f"Error during LGBMClassifier training: {e}")
            continue

        try:
            xgb_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during XGBoost Classifier training: {e}")
            logging.error(f"Error during XGBoost Classifier training: {e}")
            continue

        try:
            rf_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during RandomForestClassifier training: {e}")
            logging.error(f"Error during RandomForestClassifier training: {e}")
            continue

        # meta-features, classification
        fold_meta_features_clf = generate_meta_features_clf(
            fcnn1=fcnn1_clf,
            fcnn2=fcnn2_clf,
            lgbm_clf=lgbm_clf,
            xgb_clf=xgb_clf,
            rf_clf=rf_clf,
            test_loader=test_loader_clf,
            X_test_scaled=X_test_clf_scaled,
            device=device
        )

        meta_train_features_clf_cv2.append(fold_meta_features_clf)
        meta_train_labels_clf_cv2.extend(y_test_clf_fold.values)

    # combine regression meta-features (first CV)
    meta_train_features_reg_cv1 = np.vstack(meta_train_features_reg_cv1)
    meta_train_labels_reg_cv1 = np.array(meta_train_labels_reg_cv1)

    # combine classification meta-features (first CV)
    meta_train_features_clf_cv1 = np.vstack(meta_train_features_clf_cv1)
    meta_train_labels_clf_cv1 = np.array(meta_train_labels_clf_cv1)

    # regression meta model
    print("\nTraining Meta-Model for Regression...")
    logging.info("Training Meta-Model for Regression started.")
    meta_model_reg = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.15,
        max_depth=5,
        random_state=42
    )
    meta_model_reg.fit(meta_train_features_reg_cv1, meta_train_labels_reg_cv1)

    # classification meta model
    print("\nTraining Meta-Model for Classification...")
    logging.info("Training Meta-Model for Classification started.")
    meta_model_clf = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.15,
        max_depth=5,
        random_state=42
    )
    meta_model_clf.fit(meta_train_features_clf_cv1, meta_train_labels_clf_cv1)

    # second CV: evaluate the ensemble
    print("\nEvaluating Ensemble Models for Regression and Classification...")
    logging.info("Evaluating Ensemble Models started.")

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\nFold {fold + 1} - Evaluating Ensemble Models")
        logging.info(f"Fold {fold + 1} - Evaluating Ensemble Models started")

        X_train_reg, X_test_reg = X.iloc[train_index], X.iloc[test_index]
        y_train_reg_fold, y_test_reg_fold = y_reg.iloc[train_index], y_reg.iloc[test_index]

        X_train_clf, X_test_clf = X.iloc[train_index], X.iloc[test_index]
        y_train_clf_fold, y_test_clf_fold = y_class.iloc[train_index], y_class.iloc[test_index]

        X_train_reg_scaled, X_test_reg_scaled, scaler_X_reg_eval = preprocess_and_scale(
            X_train_reg, X_test_reg, scaler_type='StandardScaler'
        )
        y_train_reg_scaled, y_test_reg_scaled, scaler_y_reg_eval = preprocess_and_scale_reg_target(
            y_train_reg_fold, y_test_reg_fold, scaler_type='StandardScaler'
        )
        X_train_clf_scaled, X_test_clf_scaled, scaler_X_clf_eval = preprocess_and_scale(
            X_train_clf, X_test_clf, scaler_type='StandardScaler'
        )

        train_dataset_reg = FinancialRegressionDataset(X_train_reg_scaled, y_train_reg_scaled)
        test_dataset_reg = FinancialRegressionDataset(X_test_reg_scaled, y_test_reg_scaled)
        train_loader_reg = DataLoader(train_dataset_reg, batch_size=32, shuffle=True)
        test_loader_reg = DataLoader(test_dataset_reg, batch_size=32, shuffle=False)

        train_dataset_clf = FinancialClassificationDataset(X_train_clf_scaled, y_train_clf_fold)
        test_dataset_clf = FinancialClassificationDataset(X_test_clf_scaled, y_test_clf_fold)
        train_loader_clf = DataLoader(train_dataset_clf, batch_size=32, shuffle=True)
        test_loader_clf = DataLoader(test_dataset_clf, batch_size=32, shuffle=False)

        # base models, regression
        models_reg, optimizers_reg, schedulers_reg, early_stoppings_reg, mse_loss = train_base_models_reg(
            X_train_reg_scaled, y_train_reg_scaled, device
        )
        loss_history1_reg, loss_history2_reg = train_fcnn_reg(
            models=models_reg,
            optimizers=optimizers_reg,
            schedulers=schedulers_reg,
            early_stoppings=early_stoppings_reg,
            mse_loss=mse_loss,
            train_loader=train_loader_reg,
            device=device
        )

        plot_loss_curve(loss_history1_reg, loss_history2_reg, fold, task='regression', eval=True)

        fcnn1_reg, fcnn2_reg, lgbm_reg, xgb_reg, rf_reg = models_reg

        try:
            lgbm_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during LGBMRegressor training: {e}")
            logging.error(f"Error during LGBMRegressor training: {e}")
            continue

        try:
            xgb_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during XGBoost Regressor training: {e}")
            logging.error(f"Error during XGBoost Regressor training: {e}")
            continue

        try:
            rf_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during RandomForestRegressor training: {e}")
            logging.error(f"Error during RandomForestRegressor training: {e}")
            continue

        # meta-features, regression
        fold_meta_features_reg = generate_meta_features_reg(
            fcnn1=fcnn1_reg,
            fcnn2=fcnn2_reg,
            lgbm_reg=lgbm_reg,
            xgb_reg=xgb_reg,
            rf_reg=rf_reg,
            test_loader=test_loader_reg,
            X_test_scaled=X_test_reg_scaled,
            device=device
        )

        fold_meta_preds_reg_scaled = meta_model_reg.predict(fold_meta_features_reg)
        # back to raw prices for this fold
        fold_meta_preds_reg, rmse, mape = regression_metrics_original_price(
            y_test_reg_fold,
            fold_meta_preds_reg_scaled,
            prediction_scaler=scaler_y_reg_eval,
        )

        rmses.append(rmse)
        mapes.append(mape)

        print(f"Fold {fold + 1} - Regression: RMSE (USD/oz)={rmse:.4f}, MAPE (%)={mape:.4f}")
        logging.info(f"Fold {fold + 1} - Regression: RMSE (USD/oz)={rmse:.4f}, MAPE (%)={mape:.4f}")

        # base models, classification
        models_clf, optimizers_clf, schedulers_clf, early_stoppings_clf, focal_loss = train_base_models_clf(
            X_train_clf_scaled, y_train_clf_fold, device
        )
        loss_history1_clf, loss_history2_clf = train_fcnn_clf(
            models=models_clf,
            optimizers=optimizers_clf,
            schedulers=schedulers_clf,
            early_stoppings=early_stoppings_clf,
            focal_loss=focal_loss,
            train_loader=train_loader_clf,
            device=device
        )

        plot_loss_curve(loss_history1_clf, loss_history2_clf, fold, task='classification', eval=True)

        fcnn1_clf, fcnn2_clf, lgbm_clf, xgb_clf, rf_clf = models_clf

        try:
            lgbm_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during LGBMClassifier training: {e}")
            logging.error(f"Error during LGBMClassifier training: {e}")
            continue

        try:
            xgb_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during XGBoost Classifier training: {e}")
            logging.error(f"Error during XGBoost Classifier training: {e}")
            continue

        try:
            rf_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during RandomForestClassifier training: {e}")
            logging.error(f"Error during RandomForestClassifier training: {e}")
            continue

        # meta-features, classification
        fold_meta_features_clf = generate_meta_features_clf(
            fcnn1=fcnn1_clf,
            fcnn2=fcnn2_clf,
            lgbm_clf=lgbm_clf,
            xgb_clf=xgb_clf,
            rf_clf=rf_clf,
            test_loader=test_loader_clf,
            X_test_scaled=X_test_clf_scaled,
            device=device
        )

        meta_train_features_clf_cv2.append(fold_meta_features_clf)
        meta_train_labels_clf_cv2.extend(y_test_clf_fold.values)

    # combine regression meta-features (first CV)
    meta_train_features_reg_cv1 = np.vstack(meta_train_features_reg_cv1)
    meta_train_labels_reg_cv1 = np.array(meta_train_labels_reg_cv1)

    # combine classification meta-features (first CV)
    meta_train_features_clf_cv1 = np.vstack(meta_train_features_clf_cv1)
    meta_train_labels_clf_cv1 = np.array(meta_train_labels_clf_cv1)

    # regression meta model
    print("\nTraining Meta-Model for Regression...")
    logging.info("Training Meta-Model for Regression started.")
    meta_model_reg = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.15,
        max_depth=5,
        random_state=42
    )
    meta_model_reg.fit(meta_train_features_reg_cv1, meta_train_labels_reg_cv1)

    # classification meta model
    print("\nTraining Meta-Model for Classification...")
    logging.info("Training Meta-Model for Classification started.")
    meta_model_clf = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.15,
        max_depth=5,
        random_state=42
    )
    meta_model_clf.fit(meta_train_features_clf_cv1, meta_train_labels_clf_cv1)

    # evaluate
    print("\nOverall Regression Results:")
    print(f"Average RMSE (USD/oz): {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"Average MAPE (%): {np.mean(mapes):.4f} ± {np.std(mapes):.4f}")
    logging.info(
        f"Overall Regression Results: "
        f"Average RMSE (USD/oz)={np.mean(rmses):.4f} ± {np.std(rmses):.4f}, "
        f"Average MAPE (%)={np.mean(mapes):.4f} ± {np.std(mapes):.4f}"
    )

    print("\nOverall Classification Results:")
    print(f"Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Average Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"Average Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"Average F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"Average AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    logging.info(
        f"Overall Classification Results: "
        f"Average Accuracy={np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}, "
        f"Average Precision={np.mean(precisions):.4f} ± {np.std(precisions):.4f}, "
        f"Average Recall={np.mean(recalls):.4f} ± {np.std(recalls):.4f}, "
        f"Average F1 Score={np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}, "
        f"Average AUC={np.mean(aucs):.4f} ± {np.std(aucs):.4f}"
    )

    # overall results
    with open('evaluation_results.txt', 'w') as f:
        f.write("Overall Regression Results:\n")
        f.write(f"Average RMSE (USD/oz): {np.mean(rmses):.4f} ± {np.std(rmses):.4f}\n")
        f.write(f"Average MAPE (%): {np.mean(mapes):.4f} ± {np.std(mapes):.4f}\n\n")
        f.write("Overall Classification Results:\n")
        f.write(f"Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n")
        f.write(f"Average Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}\n")
        f.write(f"Average Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}\n")
        f.write(f"Average F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}\n")
        f.write(f"Average AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}\n")

if __name__ == "__main__":
    main()
