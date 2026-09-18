import pandas as pd
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
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor
)
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from torch.utils.data import Dataset, DataLoader
import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ta  # Ensure ta library is installed: pip install ta
import warnings
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import autocast, GradScaler
import optuna
import sys
import joblib

warnings.filterwarnings('ignore')  # Suppress warnings

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configure logging
logging.basicConfig(
    filename='training_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define the feature list
original_features = [
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


def generate_meta_features_reg(fcnn2, lgbm_reg, xgb_reg, rf_reg, test_loader, X_test_scaled, device):
    try:
        # Ensure fcnn2 is in eval mode
        fcnn2.eval()
        preds2 = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                output2 = fcnn2(X_batch).squeeze().cpu().numpy()
                preds2.extend(output2)

        # LightGBM predictions
        lgbm_preds = lgbm_reg.predict(X_test_scaled)

        # XGBoost predictions
        xgb_preds = xgb_reg.predict(X_test_scaled)

        # Random Forest predictions
        rf_preds = rf_reg.predict(X_test_scaled)

        # Stack all predictions as meta-features
        fold_meta_features = np.column_stack([
            preds2,
            lgbm_preds,
            xgb_preds,
            rf_preds
        ])

        # Define meta feature names for Regression
        meta_feature_names_reg = [
            'FCNN_pred',
            'LGBM_pred',
            'XGB_pred',
            'RF_pred'
        ]

        # Verify that all models are fitted
        assert hasattr(lgbm_reg, 'predict'), "lgbm_reg is not fitted."
        assert hasattr(xgb_reg, 'predict'), "xgb_reg is not fitted."
        assert hasattr(rf_reg, 'predict'), "rf_reg is not fitted."

        logging.info("Meta-features for Regression generated successfully.")
        return fold_meta_features, meta_feature_names_reg
    except AssertionError as ae:
        logging.error(f"Model fitting issue: {ae}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error in generate_meta_features_reg: {e}")
        sys.exit(1)


def generate_meta_features_clf(fcnn2, lgbm_clf, xgb_clf, rf_clf, test_loader, X_test_scaled, device):
    try:
        # Ensure fcnn2 is in eval mode
        fcnn2.eval()
        preds2 = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                output2 = fcnn2(X_batch)
                probs = torch.softmax(output2, dim=1)[:, 1].cpu().numpy()  # Only class 1 probability
                preds2.extend(probs)

        # LightGBM predictions: class 1 probabilities
        lgbm_probs = lgbm_clf.predict_proba(X_test_scaled)[:, 1]

        # XGBoost predictions: class 1 probabilities
        xgb_probs = xgb_clf.predict_proba(X_test_scaled)[:, 1]

        # Random Forest predictions: class 1 probabilities
        rf_probs = rf_clf.predict_proba(X_test_scaled)[:, 1]

        # Stack all predictions as meta-features
        fold_meta_features = np.column_stack([
            preds2,
            lgbm_probs,
            xgb_probs,
            rf_probs
        ])

        # Define meta feature names for Classification
        meta_feature_names_clf = [
            'FCNN_proba_class1',
            'LGBM_proba',
            'XGB_proba',
            'RF_proba'
        ]

        # Verify that all models are fitted
        assert hasattr(lgbm_clf, 'predict_proba'), "lgbm_clf is not fitted."
        assert hasattr(xgb_clf, 'predict_proba'), "xgb_clf is not fitted."
        assert hasattr(rf_clf, 'predict_proba'), "rf_clf is not fitted."

        logging.info("Meta-features for Classification generated successfully.")
        return fold_meta_features, meta_feature_names_clf
    except AssertionError as ae:
        logging.error(f"Model fitting issue: {ae}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error in generate_meta_features_clf: {e}")
        sys.exit(1)


# Define the Dataset class for Regression
class FinancialRegressionDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets.values.astype(np.float32)  # Ensure targets are float for regression

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )


# Define the Dataset class for Classification
class FinancialClassificationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels.values.astype(np.int64)  # Replace np.long with np.int64

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


# Define Early Stopping
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


# Define Focal Loss for Classification (Optional)
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)  # Raw logits
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


# Define the Fully Connected Neural Network Model for Regression
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
        return self.network(x)  # Raw continuous output


# Define the Fully Connected Neural Network Model for Classification
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
        return self.network(x)  # Raw logits


# Define RMS (Root Mean Squared Error)
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Define MAPE (Mean Absolute Percentage Error)
def mean_absolute_percentage_error_custom(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)


# Define data preprocessing function
def preprocess_data(file_path):
    try:
        # Load data
        data = pd.read_excel(file_path)

        # Ensure 'Date' column is datetime
        if not np.issubdtype(data['Date'].dtype, np.datetime64):
            data['Date'] = pd.to_datetime(data['Date'])

        # Create Classification Target: 1 if next day's GOLD price increases, else 0
        data['Next_Day_Change'] = (data['GOLD'].diff() > 0).astype(int)

        # Create Regression Target: next day's GOLD price
        data['Next_Day_Price'] = data['GOLD'].shift(-1)

        # Feature engineering
        # Lag features
        for lag in range(1, 31):
            data[f'GOLD_lag{lag}'] = data['GOLD'].shift(lag)

        # Moving averages
        data['GOLD_MA_3_days'] = data['GOLD'].rolling(window=3).mean()
        data['GOLD_MA_5_days'] = data['GOLD'].rolling(window=5).mean()
        data['GOLD_MA_10_days'] = data['GOLD'].rolling(window=10).mean()

        # Technical indicators
        data['RSI'] = ta.momentum.RSIIndicator(close=data['GOLD'], window=14).rsi()
        macd = ta.trend.MACD(close=data['GOLD'])
        data['MACD'] = macd.macd_diff()

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=data['GOLD'], window=20)
        data['Bollinger_High'] = bollinger.bollinger_hband()
        data['Bollinger_Low'] = bollinger.bollinger_lband()

        # Rolling statistics
        rolling_window = 10
        data['GOLD_roll_mean_5'] = data['GOLD'].rolling(window=rolling_window).mean()
        data['GOLD_roll_std_5'] = data['GOLD'].rolling(window=rolling_window).std()
        data['GOLD_roll_min_5'] = data['GOLD'].rolling(window=rolling_window).min()
        data['GOLD_roll_max_5'] = data['GOLD'].rolling(window=rolling_window).max()

        # Custom features
        data['Gold_Oil_Ratio'] = data['GOLD'] / data['CrudeOil_SpotPrice_BrentUK']
        data['USD_CNY_to_JPY'] = data['SpotRate_USD_CNY'] / data['SpotRate_Tokyo_9AM_USD_JPY']
        data['China_US_CPI_Ratio'] = data['China_CPI_YoY_CurrentMonth'] / data['US_CPI_YoY_NSA']

        # Rate of change features
        data['Gold_Rate_of_Change'] = data['GOLD'].pct_change(periods=5)
        data['Oil_Rate_of_Change'] = data['CrudeOil_SpotPrice_BrentUK'].pct_change(periods=5)

        # Trend feature
        data['Gold_Trend_7_days'] = data['GOLD'].rolling(window=7).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 7 else np.nan, raw=True
        )

        # Inflation-adjusted gold price
        data['Gold_Inflation_Adjusted'] = data['GOLD'] / data['US_CPI_YoY_NSA']

        # Volatility features
        data['Gold_Volatility_10_days'] = data['GOLD'].rolling(window=10).std()
        data['Oil_Volatility_10_days'] = data['CrudeOil_SpotPrice_BrentUK'].rolling(window=10).std()

        # Interaction features
        data['Gold_SP500_Ratio'] = data['GOLD'] / data['US_SP500_Index']
        data['US_Japan_Interest_Rate_Diff'] = data['US_DowJones_IndustrialAverage'] - data['SpotRate_Tokyo_9AM_USD_JPY']

        # Time-related features
        data['Month'] = data['Date'].dt.month
        data['Quarter'] = data['Date'].dt.quarter
        data['Day_of_Week'] = data['Date'].dt.dayofweek

        # Shift today features by 1 day
        today_features = [
            'Gold_Oil_Ratio', 'USD_CNY_to_JPY', 'China_US_CPI_Ratio',
            'Gold_Rate_of_Change', 'Oil_Rate_of_Change', 'Gold_Trend_7_days',
            'Gold_Inflation_Adjusted', 'Gold_Volatility_10_days', 'Oil_Volatility_10_days',
            'Gold_SP500_Ratio', 'US_Japan_Interest_Rate_Diff'
        ]
        data[today_features] = data[today_features].shift(1)

        # Clean data: remove NaN or inf
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        # Define feature list
        features = original_features.copy()

        # Check if all features are present
        missing_features = [feature for feature in features if feature not in data.columns]
        if missing_features:
            raise ValueError(f"The following required features are missing from the data: {missing_features}")

        # Split features and targets
        X = data[features]
        y_class = data['Next_Day_Change']
        y_reg = data['Next_Day_Price']

        logging.info("Data preprocessing completed successfully.")
        return X, y_class, y_reg

    except Exception as e:
        print(f"An error occurred during data preprocessing: {e}")
        logging.error(f"Error in preprocess_data: {e}")
        sys.exit(1)


# Feature scaling function
def preprocess_and_scale(X_train, X_test, scaler_type='StandardScaler'):
    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_type == 'RobustScaler':
        scaler = RobustScaler()
    else:
        raise ValueError("Unsupported scaler type. Choose 'StandardScaler' or 'RobustScaler'.")

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


# LightGBM objective for Regression
def objective_lgbm_reg(trial, X_train, y_train):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
        'min_gain_to_split': trial.suggest_loguniform('min_gain_to_split', 0.001, 1.0),
        'random_state': 42,
        'n_jobs': -1
    }

    lgbm = LGBMRegressor(**param)
    score = cross_val_score(lgbm, X_train, y_train, cv=3, scoring='neg_mean_squared_error').mean()
    return -score  # We want to minimize MSE


# Optimize LightGBM for Regression
def optimize_lgbm_reg(X_train, y_train):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_lgbm_reg(trial, X_train, y_train), n_trials=50)

    print("Best parameters for LGBMRegressor:", study.best_params)
    print("Best MSE for LGBMRegressor:", study.best_value)
    logging.info(f"Best parameters for LGBMRegressor: {study.best_params}")
    logging.info(f"Best MSE for LGBMRegressor: {study.best_value}")

    return study.best_params


# LightGBM objective for Classification
def objective_lgbm_clf(trial, X_train, y_train):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
        'min_gain_to_split': trial.suggest_loguniform('min_gain_to_split', 0.001, 1.0),
        'random_state': 42,
        'n_jobs': -1
    }

    lgbm = LGBMClassifier(**param)
    score = cross_val_score(lgbm, X_train, y_train, cv=3, scoring='accuracy').mean()
    return score  # We want to maximize Accuracy


# Optimize LightGBM for Classification
def optimize_lgbm_clf(X_train, y_train):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_lgbm_clf(trial, X_train, y_train), n_trials=50)

    print("Best parameters for LGBMClassifier:", study.best_params)
    print("Best Accuracy for LGBMClassifier:", study.best_value)
    logging.info(f"Best parameters for LGBMClassifier: {study.best_params}")
    logging.info(f"Best Accuracy for LGBMClassifier: {study.best_value}")

    return study.best_params


# Train base models for Regression
def train_base_models_reg(X_train, y_train, device):
    # Initialize FCNN Regressor
    hidden_dims = [512, 256, 128, 64]
    fcnn2 = FCNNRegressor(input_dim=X_train.shape[1], hidden_dims=hidden_dims).to(device)

    # Optimize and initialize LightGBM Regressor
    best_params_lgbm = optimize_lgbm_reg(X_train, y_train)
    lgbm_reg = LGBMRegressor(**best_params_lgbm, random_state=42, n_jobs=-1)

    # Initialize XGBoost Regressor
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

    # Initialize Random Forest Regressor
    rf_reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    # Initialize optimizer with AdamW
    optimizer2 = optim.AdamW(fcnn2.parameters(), lr=0.005, weight_decay=1e-5)

    # Initialize learning rate scheduler with Cosine Annealing
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=50)

    # Initialize Early Stopping
    early_stopping2 = EarlyStopping(patience=12)

    # Initialize MSE Loss
    mse_loss = nn.MSELoss()

    # Group all models and related components
    models = (fcnn2, lgbm_reg, xgb_reg, rf_reg)
    optimizers = (optimizer2,)
    schedulers = (scheduler2,)
    early_stoppings = (early_stopping2,)

    return models, optimizers, schedulers, early_stoppings, mse_loss


# Train base models for Classification
def train_base_models_clf(X_train, y_train, device):
    # Initialize FCNN Classifier
    hidden_dims = [512, 256, 128, 64]
    fcnn2 = FCNNClassifier(input_dim=X_train.shape[1], hidden_dims=hidden_dims).to(device)

    # Optimize and initialize LightGBM Classifier
    best_params_lgbm = optimize_lgbm_clf(X_train, y_train)
    lgbm_clf = LGBMClassifier(**best_params_lgbm, random_state=42, n_jobs=-1)

    # Initialize XGBoost Classifier
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

    # Initialize Random Forest Classifier
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    # Initialize optimizer with AdamW
    optimizer2 = optim.AdamW(fcnn2.parameters(), lr=0.0005, weight_decay=1e-5)

    # Initialize learning rate scheduler with Cosine Annealing
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=50)

    # Initialize Early Stopping
    early_stopping2 = EarlyStopping(patience=15)

    # Compute class weights and initialize Focal Loss
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    focal_loss = FocalLoss(alpha=class_weights, gamma=2)

    # Group all models and related components
    models = (fcnn2, lgbm_clf, xgb_clf, rf_clf)
    optimizers = (optimizer2,)
    schedulers = (scheduler2,)
    early_stoppings = (early_stopping2,)

    return models, optimizers, schedulers, early_stoppings, focal_loss


# Train FCNN Regressor
def train_fcnn_reg(models, optimizers, schedulers, early_stoppings, mse_loss, train_loader, device, num_epochs=300):
    fcnn2, _, _, _ = models
    optimizer2, = optimizers
    scheduler2, = schedulers
    early_stopping2, = early_stoppings

    loss_history2 = []
    scaler = GradScaler()

    for epoch in range(num_epochs):
        fcnn2.train()
        running_loss2 = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Train FCNN2
            optimizer2.zero_grad()
            with autocast():
                outputs2 = fcnn2(X_batch).squeeze()
                loss2 = mse_loss(outputs2, y_batch)
            scaler.scale(loss2).backward()
            scaler.step(optimizer2)
            scaler.update()
            running_loss2 += loss2.item()

        # Calculate average loss
        avg_loss2 = running_loss2 / len(train_loader)
        loss_history2.append(avg_loss2)

        # Print and log loss
        print(f"Epoch {epoch + 1} - FCNN2 Loss: {avg_loss2:.4f}")
        logging.info(f"Epoch {epoch + 1} - FCNN2 Loss: {avg_loss2:.4f}")

        # Step the scheduler
        scheduler2.step()

        # Check early stopping
        early_stopping2(avg_loss2)
        if early_stopping2.early_stop:
            print("Early stopping triggered.")
            logging.info("Early stopping triggered.")
            break

    return loss_history2,


# Train FCNN Classifier
def train_fcnn_clf(models, optimizers, schedulers, early_stoppings, focal_loss, train_loader, device, num_epochs=300):
    fcnn2, _, _, _ = models
    optimizer2, = optimizers
    scheduler2, = schedulers
    early_stopping2, = early_stoppings

    loss_history2 = []
    scaler = GradScaler()

    for epoch in range(num_epochs):
        fcnn2.train()
        running_loss2 = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Train FCNN2
            optimizer2.zero_grad()
            with autocast():
                outputs2 = fcnn2(X_batch)
                loss2 = focal_loss(outputs2, y_batch)
            scaler.scale(loss2).backward()
            scaler.step(optimizer2)
            scaler.update()
            running_loss2 += loss2.item()

        # Calculate average loss
        avg_loss2 = running_loss2 / len(train_loader)
        loss_history2.append(avg_loss2)

        # Print and log loss
        print(f"Epoch {epoch + 1} - FCNN2 Loss: {avg_loss2:.4f}")
        logging.info(f"Epoch {epoch + 1} - FCNN2 Loss: {avg_loss2:.4f}")

        # Step the scheduler
        scheduler2.step()

        # Check early stopping
        early_stopping2(avg_loss2)
        if early_stopping2.early_stop:
            print("Early stopping triggered.")
            logging.info("Early stopping triggered.")
            break

    return loss_history2,


# Plot loss curves
def plot_loss_curve(loss_history2, fold, task='regression', eval=False):
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history2, label='FCNN2 Loss', color='blue', linestyle='-')
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
        plt.savefig(filename)  # Save the plot as a file
        plt.close()  # Close the plot to avoid display issues
        logging.info(f"Loss curve for fold {fold + 1}, task {task} saved as {filename}.")
    except Exception as e:
        logging.error(f"Error in plot_loss_curve: {e}")
        sys.exit(1)


# Feature Ablation Analysis for Regression
def feature_ablation_regression(meta_model_reg, X_test_meta, y_test_reg, feature_names):
    try:
        # Calculate baseline RMSE
        baseline_preds = meta_model_reg.predict(X_test_meta)
        baseline_rmse = root_mean_squared_error(y_test_reg, baseline_preds)

        feature_importances = []

        # Iterate over each feature for ablation
        for i, feature in enumerate(feature_names):
            # Remove the current feature
            X_modified = np.delete(X_test_meta, i, axis=1)

            # Retrain a new regression model on the modified data
            ablation_model = GradientBoostingRegressor(
                n_estimators=meta_model_reg.n_estimators,
                learning_rate=meta_model_reg.learning_rate,
                max_depth=meta_model_reg.max_depth,
                random_state=meta_model_reg.random_state
            )
            ablation_model.fit(X_modified, y_test_reg)

            # Calculate RMSE after ablation
            preds_modified = ablation_model.predict(X_modified)
            rmse_modified = root_mean_squared_error(y_test_reg, preds_modified)

            # Importance is the increase in RMSE
            importance = rmse_modified - baseline_rmse
            feature_importances.append(importance)

        # Create DataFrame for feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        })
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        logging.info("Feature ablation for regression completed successfully.")
        return feature_importance_df
    except Exception as e:
        logging.error(f"Error in feature_ablation_regression: {e}")
        sys.exit(1)


# Feature Ablation Analysis for Classification
def feature_ablation_classification(meta_model_clf, X_test_meta, y_test_clf, feature_names):
    try:
        # Calculate baseline Accuracy
        baseline_accuracy = accuracy_score(y_test_clf, meta_model_clf.predict(X_test_meta))

        feature_importances = []

        # Iterate over each feature for ablation
        for i, feature in enumerate(feature_names):
            # Remove the current feature
            X_modified = np.delete(X_test_meta, i, axis=1)

            # Retrain a new classification model on the modified data
            ablation_model = GradientBoostingClassifier(
                n_estimators=meta_model_clf.n_estimators,
                learning_rate=meta_model_clf.learning_rate,
                max_depth=meta_model_clf.max_depth,
                random_state=meta_model_clf.random_state
            )
            ablation_model.fit(X_modified, y_test_clf)

            # Calculate Accuracy after ablation
            preds_modified = ablation_model.predict(X_modified)
            accuracy_modified = accuracy_score(y_test_clf, preds_modified)

            # Importance is the decrease in Accuracy
            importance = baseline_accuracy - accuracy_modified
            feature_importances.append(importance)

        # Create DataFrame for feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        })
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        logging.info("Feature ablation for classification completed successfully.")
        return feature_importance_df
    except Exception as e:
        logging.error(f"Error in feature_ablation_classification: {e}")
        sys.exit(1)


# Plot Feature Importance
def plot_feature_importance(feature_importance_df, task='regression'):
    try:
        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

        plt.figure(figsize=(10, max(6, 0.3 * len(feature_importance_df))))  # Adjust height based on number of features

        # Define color map
        cmap = mcolors.LinearSegmentedColormap.from_list("blue_orange", ["#2D6CAB", "#F17D56"])
        colors = cmap(np.linspace(0, 1, len(feature_importance_df)))

        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color=colors)
        plt.xlabel('Importance')
        plt.title(f'Feature Importance ({task.capitalize()})')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{task}.png')
        plt.close()

        logging.info(f"Feature importance plot for {task} saved successfully.")
    except Exception as e:
        logging.error(f"Error in plot_feature_importance: {e}")
        sys.exit(1)


# Feature Ablation Analysis for Original Features
def feature_ablation_analysis(X, y_class, y_reg, feature_list, device, scaler_type='StandardScaler'):
    import copy
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_percentage_error
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, \
        RandomForestClassifier
    import logging

    # Initialize TimeSeriesSplit
    tscv_ablation = TimeSeriesSplit(n_splits=5)

    # Store baseline performance
    baseline_metrics = {
        'regression': {'RMSE': [], 'MAPE': []},
        'classification': {'Accuracy': []}
    }

    # Initialize list to store feature importance
    feature_importance = pd.DataFrame(
        columns=['Feature', 'Regression_RMSE_Impact', 'Regression_MAPE_Impact', 'Classification_Accuracy_Impact'])

    # Calculate baseline performance
    print("Calculating baseline performance with all features...")
    logging.info("Calculating baseline performance with all features...")
    for fold, (train_index, test_index) in enumerate(tscv_ablation.split(X)):
        print(f"Fold {fold + 1} - Baseline Training")
        logging.info(f"Fold {fold + 1} - Baseline Training")

        # Split data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train_reg, y_test_reg = y_reg.iloc[train_index], y_reg.iloc[test_index]
        y_train_clf, y_test_clf = y_class.iloc[train_index], y_class.iloc[test_index]

        # Scale features
        X_train_scaled, X_test_scaled = preprocess_and_scale(X_train, X_test, scaler_type=scaler_type)

        # Train base models for Regression
        models_reg, optimizers_reg, schedulers_reg, early_stoppings_reg, mse_loss = train_base_models_reg(
            X_train_scaled, y_train_reg, device
        )
        loss_history2_reg, = train_fcnn_reg(
            models=models_reg,
            optimizers=optimizers_reg,
            schedulers=schedulers_reg,
            early_stoppings=early_stoppings_reg,
            mse_loss=mse_loss,
            train_loader=DataLoader(FinancialRegressionDataset(X_train_scaled, y_train_reg), batch_size=32,
                                    shuffle=True),
            device=device
        )

        # Train base models for Classification
        models_clf, optimizers_clf, schedulers_clf, early_stoppings_clf, focal_loss = train_base_models_clf(
            X_train_scaled, y_train_clf, device
        )
        loss_history2_clf, = train_fcnn_clf(
            models=models_clf,
            optimizers=optimizers_clf,
            schedulers=schedulers_clf,
            early_stoppings=early_stoppings_clf,
            focal_loss=focal_loss,
            train_loader=DataLoader(FinancialClassificationDataset(X_train_scaled, y_train_clf), batch_size=32,
                                    shuffle=True),
            device=device
        )

        # Get Regression Models
        fcnn2_reg, lgbm_reg, xgb_reg, rf_reg = models_reg

        # Get Classification Models
        fcnn2_clf, lgbm_clf, xgb_clf, rf_clf = models_clf

        # Train LightGBM Regressor (Already optimized and trained in train_base_models_reg)
        try:
            lgbm_reg.fit(X_train_scaled, y_train_reg)
            logging.info(f"Fold {fold + 1} - LGBMRegressor trained successfully.")
        except Exception as e:
            print(f"Error during LGBMRegressor training: {e}")
            logging.error(f"Error during LGBMRegressor training: {e}")
            sys.exit(1)

        # Train XGBoost Regressor (Already initialized and trained in train_base_models_reg)
        try:
            xgb_reg.fit(X_train_scaled, y_train_reg)
            logging.info(f"Fold {fold + 1} - XGBoostRegressor trained successfully.")
        except Exception as e:
            print(f"Error during XGBoost Regressor training: {e}")
            logging.error(f"Error during XGBoost Regressor training: {e}")
            sys.exit(1)

        # Train Random Forest Regressor (Already initialized and trained in train_base_models_reg)
        try:
            rf_reg.fit(X_train_scaled, y_train_reg)
            logging.info(f"Fold {fold + 1} - RandomForestRegressor trained successfully.")
        except Exception as e:
            print(f"Error during RandomForestRegressor training: {e}")
            logging.error(f"Error during RandomForestRegressor training: {e}")
            sys.exit(1)

        # Generate meta-features for Regression
        fold_meta_features_reg, meta_feature_names_reg = generate_meta_features_reg(
            fcnn2=fcnn2_reg,
            lgbm_reg=lgbm_reg,
            xgb_reg=xgb_reg,
            rf_reg=rf_reg,
            test_loader=DataLoader(FinancialRegressionDataset(X_test_scaled, y_test_reg), batch_size=32, shuffle=False),
            X_test_scaled=X_test_scaled,
            device=device
        )

        # Train LightGBM Classifier (Already optimized and trained in train_base_models_clf)
        try:
            lgbm_clf.fit(X_train_scaled, y_train_clf)
            logging.info(f"Fold {fold + 1} - LGBMClassifier trained successfully.")
        except Exception as e:
            print(f"Error during LGBMClassifier training: {e}")
            logging.error(f"Error during LGBMClassifier training: {e}")
            sys.exit(1)

        # Train XGBoost Classifier (Already initialized and trained in train_base_models_clf)
        try:
            xgb_clf.fit(X_train_scaled, y_train_clf)
            logging.info(f"Fold {fold + 1} - XGBoostClassifier trained successfully.")
        except Exception as e:
            print(f"Error during XGBoost Classifier training: {e}")
            logging.error(f"Error during XGBoost Classifier training: {e}")
            sys.exit(1)

        # Train Random Forest Classifier (Already initialized and trained in train_base_models_clf)
        try:
            rf_clf.fit(X_train_scaled, y_train_clf)
            logging.info(f"Fold {fold + 1} - RandomForestClassifier trained successfully.")
        except Exception as e:
            print(f"Error during RandomForestClassifier training: {e}")
            logging.error(f"Error during RandomForestClassifier training: {e}")
            sys.exit(1)

        # Generate meta-features for Classification
        fold_meta_features_clf, meta_feature_names_clf = generate_meta_features_clf(
            fcnn2=fcnn2_clf,
            lgbm_clf=lgbm_clf,
            xgb_clf=xgb_clf,
            rf_clf=rf_clf,
            test_loader=DataLoader(FinancialClassificationDataset(X_test_scaled, y_test_clf), batch_size=32,
                                   shuffle=False),
            X_test_scaled=X_test_scaled,
            device=device
        )

        # Combine meta-features
        meta_features_reg = fold_meta_features_reg
        meta_labels_reg = y_test_reg.values

        meta_features_clf = fold_meta_features_clf
        meta_labels_clf = y_test_clf.values

        # Train Meta-Model for Regression
        meta_model_reg = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.15,
            max_depth=5,
            random_state=42
        )
        meta_model_reg.fit(meta_features_reg, meta_labels_reg)

        # Train Meta-Model for Classification
        meta_model_clf = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.15,
            max_depth=5,
            random_state=42
        )
        meta_model_clf.fit(meta_features_clf, meta_labels_clf)

        # Evaluate Meta-Model for Regression
        y_pred_reg = meta_model_reg.predict(meta_features_reg)
        rmse = root_mean_squared_error(meta_labels_reg, y_pred_reg)
        mape = mean_absolute_percentage_error_custom(meta_labels_reg, y_pred_reg)
        baseline_metrics['regression']['RMSE'].append(rmse)
        baseline_metrics['regression']['MAPE'].append(mape)

        # Evaluate Meta-Model for Classification
        y_pred_clf = meta_model_clf.predict(meta_features_clf)
        accuracy = accuracy_score(meta_labels_clf, y_pred_clf)
        baseline_metrics['classification']['Accuracy'].append(accuracy)

    # Record baseline performance
    baseline_regression_rmse = np.mean(baseline_metrics['regression']['RMSE'])
    baseline_regression_mape = np.mean(baseline_metrics['regression']['MAPE'])
    baseline_classification_accuracy = np.mean(baseline_metrics['classification']['Accuracy'])

    print(f"Baseline Regression RMSE: {baseline_regression_rmse:.4f}")
    print(f"Baseline Regression MAPE: {baseline_regression_mape:.4f}")
    print(f"Baseline Classification Accuracy: {baseline_classification_accuracy:.4f}")

    logging.info(f"Baseline Regression RMSE: {baseline_regression_rmse:.4f}")
    logging.info(f"Baseline Regression MAPE: {baseline_regression_mape:.4f}")
    logging.info(f"Baseline Classification Accuracy: {baseline_classification_accuracy:.4f}")

    # Start feature ablation analysis
    print("\nStarting Feature Ablation Analysis...")
    logging.info("Starting Feature Ablation Analysis...")

    for feature in feature_list:
        print(f"\nAblating feature: {feature}")
        logging.info(f"Ablating feature: {feature}")

        # Remove the feature from the dataset
        X_ablation = X.drop(columns=[feature])

        # Initialize TimeSeriesSplit
        tscv_ablation_inner = TimeSeriesSplit(n_splits=5)

        # Initialize metrics
        ablation_metrics = {
            'regression': {'RMSE': [], 'MAPE': []},
            'classification': {'Accuracy': []}
        }

        for fold_inner, (train_idx_inner, test_idx_inner) in enumerate(tscv_ablation_inner.split(X_ablation)):
            print(f"Fold {fold_inner + 1} - Training with feature ablation: {feature}")
            logging.info(f"Fold {fold_inner + 1} - Training with feature ablation: {feature}")

            # Split data
            X_train_inner, X_test_inner = X_ablation.iloc[train_idx_inner], X_ablation.iloc[test_idx_inner]
            y_train_reg_inner, y_test_reg_inner = y_reg.iloc[train_idx_inner], y_reg.iloc[test_idx_inner]
            y_train_clf_inner, y_test_clf_inner = y_class.iloc[train_idx_inner], y_class.iloc[test_idx_inner]

            # Scale features
            X_train_inner_scaled, X_test_inner_scaled = preprocess_and_scale(X_train_inner, X_test_inner,
                                                                             scaler_type=scaler_type)

            # Train base models for Regression
            models_reg_inner, optimizers_reg_inner, schedulers_reg_inner, early_stoppings_reg_inner, mse_loss_inner = train_base_models_reg(
                X_train_inner_scaled, y_train_reg_inner, device
            )
            loss_history2_reg_inner, = train_fcnn_reg(
                models=models_reg_inner,
                optimizers=optimizers_reg_inner,
                schedulers=schedulers_reg_inner,
                early_stoppings=early_stoppings_reg_inner,
                mse_loss=mse_loss_inner,
                train_loader=DataLoader(FinancialRegressionDataset(X_train_inner_scaled, y_train_reg_inner), batch_size=32, shuffle=True),
                device=device
            )

            # Train base models for Classification
            models_clf_inner, optimizers_clf_inner, schedulers_clf_inner, early_stoppings_clf_inner, focal_loss_inner = train_base_models_clf(
                X_train_inner_scaled, y_train_clf_inner, device
            )
            loss_history2_clf_inner, = train_fcnn_clf(
                models=models_clf_inner,
                optimizers=optimizers_clf_inner,
                schedulers=schedulers_clf_inner,
                early_stoppings=early_stoppings_clf_inner,
                focal_loss=focal_loss_inner,
                train_loader=DataLoader(FinancialClassificationDataset(X_train_inner_scaled, y_train_clf_inner), batch_size=32, shuffle=True),
                device=device
            )

            # Get Regression Models
            fcnn2_reg_inner, lgbm_reg_inner, xgb_reg_inner, rf_reg_inner = models_reg_inner

            # Get Classification Models
            fcnn2_clf_inner, lgbm_clf_inner, xgb_clf_inner, rf_clf_inner = models_clf_inner

            # Train LightGBM Regressor (Already optimized and trained in train_base_models_reg)
            try:
                lgbm_reg_inner.fit(X_train_inner_scaled, y_train_reg_inner)
                logging.info(f"Fold {fold_inner + 1} - LGBMRegressor trained successfully.")
            except Exception as e:
                print(f"Error during LGBMRegressor training: {e}")
                logging.error(f"Error during LGBMRegressor training: {e}")
                sys.exit(1)

            # Train XGBoost Regressor (Already initialized and trained in train_base_models_reg)
            try:
                xgb_reg_inner.fit(X_train_inner_scaled, y_train_reg_inner)
                logging.info(f"Fold {fold_inner + 1} - XGBoostRegressor trained successfully.")
            except Exception as e:
                print(f"Error during XGBoost Regressor training: {e}")
                logging.error(f"Error during XGBoost Regressor training: {e}")
                sys.exit(1)

            # Train Random Forest Regressor (Already initialized and trained in train_base_models_reg)
            try:
                rf_reg_inner.fit(X_train_inner_scaled, y_train_reg_inner)
                logging.info(f"Fold {fold_inner + 1} - RandomForestRegressor trained successfully.")
            except Exception as e:
                print(f"Error during RandomForestRegressor training: {e}")
                logging.error(f"Error during RandomForestRegressor training: {e}")
                sys.exit(1)

            # Generate meta-features for Regression
            fold_meta_features_reg_inner, meta_feature_names_reg_inner = generate_meta_features_reg(
                fcnn2=fcnn2_reg_inner,
                lgbm_reg=lgbm_reg_inner,
                xgb_reg=xgb_reg_inner,
                rf_reg=rf_reg_inner,
                test_loader=DataLoader(FinancialRegressionDataset(X_test_inner_scaled, y_test_reg_inner), batch_size=32, shuffle=False),
                X_test_scaled=X_test_inner_scaled,
                device=device
            )

            # Train LightGBM Classifier (Already optimized and trained in train_base_models_clf)
            try:
                lgbm_clf_inner.fit(X_train_inner_scaled, y_train_clf_inner)
                logging.info(f"Fold {fold_inner + 1} - LGBMClassifier trained successfully.")
            except Exception as e:
                print(f"Error during LGBMClassifier training: {e}")
                logging.error(f"Error during LGBMClassifier training: {e}")
                sys.exit(1)

            # Train XGBoost Classifier (Already initialized and trained in train_base_models_clf)
            try:
                xgb_clf_inner.fit(X_train_inner_scaled, y_train_clf_inner)
                logging.info(f"Fold {fold_inner + 1} - XGBoostClassifier trained successfully.")
            except Exception as e:
                print(f"Error during XGBoost Classifier training: {e}")
                logging.error(f"Error during XGBoost Classifier training: {e}")
                sys.exit(1)

            # Train Random Forest Classifier (Already initialized and trained in train_base_models_clf)
            try:
                rf_clf_inner.fit(X_train_inner_scaled, y_train_clf_inner)
                logging.info(f"Fold {fold_inner + 1} - RandomForestClassifier trained successfully.")
            except Exception as e:
                print(f"Error during RandomForestClassifier training: {e}")
                logging.error(f"Error during RandomForestClassifier training: {e}")
                sys.exit(1)

            # Generate meta-features for Classification
            fold_meta_features_clf_inner, meta_feature_names_clf_inner = generate_meta_features_clf(
                fcnn2=fcnn2_clf_inner,
                lgbm_clf=lgbm_clf_inner,
                xgb_clf=xgb_clf_inner,
                rf_clf=rf_clf_inner,
                test_loader=DataLoader(FinancialClassificationDataset(X_test_inner_scaled, y_test_clf_inner), batch_size=32, shuffle=False),
                X_test_scaled=X_test_inner_scaled,
                device=device
            )

            # Combine meta-features
            meta_features_reg_inner = fold_meta_features_reg_inner
            meta_labels_reg_inner = y_test_reg_inner.values

            meta_features_clf_inner = fold_meta_features_clf_inner
            meta_labels_clf_inner = y_test_clf_inner.values

            # Train Meta-Model for Regression
            meta_model_reg_inner = GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.15,
                max_depth=5,
                random_state=42
            )
            meta_model_reg_inner.fit(meta_features_reg_inner, meta_labels_reg_inner)

            # Train Meta-Model for Classification
            meta_model_clf_inner = GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.15,
                max_depth=5,
                random_state=42
            )
            meta_model_clf_inner.fit(meta_features_clf_inner, meta_labels_clf_inner)

            # Evaluate Meta-Model for Regression
            y_pred_reg_inner = meta_model_reg_inner.predict(meta_features_reg_inner)
            rmse_inner = root_mean_squared_error(meta_labels_reg_inner, y_pred_reg_inner)
            mape_inner = mean_absolute_percentage_error_custom(meta_labels_reg_inner, y_pred_reg_inner)
            ablation_metrics['regression']['RMSE'].append(rmse_inner)
            ablation_metrics['regression']['MAPE'].append(mape_inner)

            # Evaluate Meta-Model for Classification
            y_pred_clf_inner = meta_model_clf_inner.predict(meta_features_clf_inner)
            accuracy_inner = accuracy_score(meta_labels_clf_inner, y_pred_clf_inner)
            ablation_metrics['classification']['Accuracy'].append(accuracy_inner)

        # Calculate average performance after ablation
        ablation_regression_rmse = np.mean(ablation_metrics['regression']['RMSE'])
        ablation_regression_mape = np.mean(ablation_metrics['regression']['MAPE'])
        ablation_classification_accuracy = np.mean(ablation_metrics['classification']['Accuracy'])

        # Calculate impact
        regression_rmse_impact = ablation_regression_rmse - baseline_regression_rmse
        regression_mape_impact = ablation_regression_mape - baseline_regression_mape
        classification_accuracy_impact = baseline_classification_accuracy - ablation_classification_accuracy

        # Append to feature importance DataFrame
        feature_importance = pd.concat([
            feature_importance,
            pd.DataFrame([{
                'Feature': feature,
                'Regression_RMSE_Impact': regression_rmse_impact,
                'Regression_MAPE_Impact': regression_mape_impact,
                'Classification_Accuracy_Impact': classification_accuracy_impact
            }])
        ], ignore_index=True)

        print(
            f"Feature: {feature} - RMSE Impact: {regression_rmse_impact:.4f}, MAPE Impact: {regression_mape_impact:.4f}, Accuracy Impact: {classification_accuracy_impact:.4f}")
        logging.info(
            f"Feature: {feature} - RMSE Impact: {regression_rmse_impact:.4f}, MAPE Impact: {regression_mape_impact:.4f}, Accuracy Impact: {classification_accuracy_impact:.4f}")

    # Save the feature importance results
    feature_importance.to_csv('feature_ablation_importance.csv', index=False)
    logging.info("Feature Ablation Analysis completed and results saved to 'feature_ablation_importance.csv'.")
    print("\nFeature Ablation Analysis completed successfully.")
    print("Feature importance results saved to 'feature_ablation_importance.csv'.")


# Plot Feature Importance Ablation Results
def plot_feature_importance_ablation(csv_path='feature_ablation_importance.csv'):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        # Read the data
        df = pd.read_csv(csv_path)

        # Regression RMSE Impact
        plt.figure(figsize=(10, 20))
        df_sorted = df.sort_values(by='Regression_RMSE_Impact', ascending=False)
        plt.barh(df_sorted['Feature'], df_sorted['Regression_RMSE_Impact'], color='skyblue')
        plt.xlabel('RMSE Impact')
        plt.title('Feature Importance based on RMSE Impact (Regression)')
        plt.tight_layout()
        plt.savefig('feature_importance_regression_ablation.png')
        plt.close()

        # Regression MAPE Impact
        plt.figure(figsize=(10, 20))
        df_sorted = df.sort_values(by='Regression_MAPE_Impact', ascending=False)
        plt.barh(df_sorted['Feature'], df_sorted['Regression_MAPE_Impact'], color='salmon')
        plt.xlabel('MAPE Impact')
        plt.title('Feature Importance based on MAPE Impact (Regression)')
        plt.tight_layout()
        plt.savefig('feature_importance_regression_mape_ablation.png')
        plt.close()

        # Classification Accuracy Impact
        plt.figure(figsize=(10, 20))
        df_sorted = df.sort_values(by='Classification_Accuracy_Impact', ascending=False)
        plt.barh(df_sorted['Feature'], df_sorted['Classification_Accuracy_Impact'], color='lightgreen')
        plt.xlabel('Accuracy Impact')
        plt.title('Feature Importance based on Accuracy Impact (Classification)')
        plt.tight_layout()
        plt.savefig('feature_importance_classification_ablation.png')
        plt.close()

        logging.info("Feature importance ablation plots generated successfully.")

        print("Feature importance ablation plots saved successfully.")

    except Exception as e:
        logging.error(f"Error in plot_feature_importance_ablation: {e}")
        print(f"Error generating plots: {e}")
        sys.exit(1)


# 主执行函数
def main():
    # File path (modify as needed)
    file_path = "/home/w/桌面/lilj/GOLD_cleaned.xlsx"
    try:
        X, y_class, y_reg = preprocess_data(file_path)
    except Exception as e:
        print(f"Error in preprocessing data: {e}")
        logging.error(f"Error in preprocessing data: {e}")
        return

    # Check target distributions
    print("Classification Target Distribution:")
    print(y_class.value_counts())
    logging.info(f"Classification Target Distribution:\n{y_class.value_counts()}")

    print("\nRegression Target Statistics:")
    print(y_reg.describe())
    logging.info(f"Regression Target Statistics:\n{y_reg.describe()}")

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    logging.info(f"Using device: {device}")

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    # Lists to store meta-features and labels
    meta_train_features_reg = []
    meta_train_labels_reg = []

    meta_train_features_clf = []
    meta_train_labels_clf = []

    # Lists to store evaluation metrics
    # For Regression
    rmses, mapes = [], []

    # For Classification
    accuracies, precisions, recalls, f1_scores, aucs = [], [], [], [], []

    # First Cross-Validation: Generate Meta-Features for Regression and Classification
    print("\nGenerating meta-features and training base models for Regression and Classification...")
    logging.info("Generating meta-features and training base models started.")

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\nFold {fold + 1} - Training base models")
        logging.info(f"Fold {fold + 1} - Training base models started")

        # Split data for Regression
        X_train_reg, X_test_reg = X.iloc[train_index], X.iloc[test_index]
        y_train_reg_fold, y_test_reg_fold = y_reg.iloc[train_index], y_reg.iloc[test_index]

        # Split data for Classification
        X_train_clf, X_test_clf = X.iloc[train_index], X.iloc[test_index]
        y_train_clf_fold, y_test_clf_fold = y_class.iloc[train_index], y_class.iloc[test_index]

        # Scale features for Regression
        X_train_reg_scaled, X_test_reg_scaled = preprocess_and_scale(X_train_reg, X_test_reg,
                                                                     scaler_type='StandardScaler')

        # Scale features for Classification
        X_train_clf_scaled, X_test_clf_scaled = preprocess_and_scale(X_train_clf, X_test_clf,
                                                                     scaler_type='StandardScaler')

        # Create Datasets and DataLoaders for Regression
        train_dataset_reg = FinancialRegressionDataset(X_train_reg_scaled, y_train_reg_fold)
        test_dataset_reg = FinancialRegressionDataset(X_test_reg_scaled, y_test_reg_fold)
        train_loader_reg = DataLoader(train_dataset_reg, batch_size=32, shuffle=True)
        test_loader_reg = DataLoader(test_dataset_reg, batch_size=32, shuffle=False)

        # Create Datasets and DataLoaders for Classification
        train_dataset_clf = FinancialClassificationDataset(X_train_clf_scaled, y_train_clf_fold)
        test_dataset_clf = FinancialClassificationDataset(X_test_clf_scaled, y_test_clf_fold)
        train_loader_clf = DataLoader(train_dataset_clf, batch_size=32, shuffle=True)
        test_loader_clf = DataLoader(test_dataset_clf, batch_size=32, shuffle=False)

        # Train base models for Regression
        models_reg, optimizers_reg, schedulers_reg, early_stoppings_reg, mse_loss = train_base_models_reg(
            X_train_reg_scaled, y_train_reg_fold, device
        )
        loss_history2_reg, = train_fcnn_reg(
            models=models_reg,
            optimizers=optimizers_reg,
            schedulers=schedulers_reg,
            early_stoppings=early_stoppings_reg,
            mse_loss=mse_loss,
            train_loader=train_loader_reg,
            device=device
        )

        # Plot loss curves for Regression
        plot_loss_curve(loss_history2_reg, fold, task='regression')

        # Get Regression Models
        fcnn2_reg, lgbm_reg, xgb_reg, rf_reg = models_reg

        # Train LightGBM Regressor (Already optimized and trained in train_base_models_reg)
        try:
            lgbm_reg.fit(X_train_reg_scaled, y_train_reg_fold)
            logging.info(f"Fold {fold + 1} - LGBMRegressor trained successfully.")
        except Exception as e:
            print(f"Error during LGBMRegressor training: {e}")
            logging.error(f"Error during LGBMRegressor training: {e}")
            sys.exit(1)

        # Train XGBoost Regressor (Already initialized and trained in train_base_models_reg)
        try:
            xgb_reg.fit(X_train_reg_scaled, y_train_reg_fold)
            logging.info(f"Fold {fold + 1} - XGBoostRegressor trained successfully.")
        except Exception as e:
            print(f"Error during XGBoost Regressor training: {e}")
            logging.error(f"Error during XGBoost Regressor training: {e}")
            sys.exit(1)

        # Train Random Forest Regressor (Already initialized and trained in train_base_models_reg)
        try:
            rf_reg.fit(X_train_reg_scaled, y_train_reg_fold)
            logging.info(f"Fold {fold + 1} - RandomForestRegressor trained successfully.")
        except Exception as e:
            print(f"Error during RandomForestRegressor training: {e}")
            logging.error(f"Error during RandomForestRegressor training: {e}")
            sys.exit(1)

        # Generate meta-features for Regression
        fold_meta_features_reg, meta_feature_names_reg = generate_meta_features_reg(
            fcnn2=fcnn2_reg,
            lgbm_reg=lgbm_reg,
            xgb_reg=xgb_reg,
            rf_reg=rf_reg,
            test_loader=test_loader_reg,
            X_test_scaled=X_test_reg_scaled,
            device=device
        )

        # Append to meta-feature lists for Regression
        meta_train_features_reg.append(fold_meta_features_reg)
        meta_train_labels_reg.extend(y_test_reg_fold.values)

        # Train base models for Classification
        models_clf, optimizers_clf, schedulers_clf, early_stoppings_clf, focal_loss = train_base_models_clf(
            X_train_clf_scaled, y_train_clf_fold, device
        )
        loss_history2_clf, = train_fcnn_clf(
            models=models_clf,
            optimizers=optimizers_clf,
            schedulers=schedulers_clf,
            early_stoppings=early_stoppings_clf,
            focal_loss=focal_loss,
            train_loader=train_loader_clf,
            device=device
        )

        # Plot loss curves for Classification
        plot_loss_curve(loss_history2_clf, fold, task='classification')

        # Get Classification Models
        fcnn2_clf, lgbm_clf, xgb_clf, rf_clf = models_clf

        # Train LightGBM Classifier (Already optimized and trained in train_base_models_clf)
        try:
            lgbm_clf.fit(X_train_clf_scaled, y_train_clf_fold)
            logging.info(f"Fold {fold + 1} - LGBMClassifier trained successfully.")
        except Exception as e:
            print(f"Error during LGBMClassifier training: {e}")
            logging.error(f"Error during LGBMClassifier training: {e}")
            sys.exit(1)

        # Train XGBoost Classifier (Already initialized and trained in train_base_models_clf)
        try:
            xgb_clf.fit(X_train_clf_scaled, y_train_clf_fold)
            logging.info(f"Fold {fold + 1} - XGBoostClassifier trained successfully.")
        except Exception as e:
            print(f"Error during XGBoost Classifier training: {e}")
            logging.error(f"Error during XGBoost Classifier training: {e}")
            sys.exit(1)

        # Train Random Forest Classifier (Already initialized and trained in train_base_models_clf)
        try:
            rf_clf.fit(X_train_clf_scaled, y_train_clf_fold)
            logging.info(f"Fold {fold + 1} - RandomForestClassifier trained successfully.")
        except Exception as e:
            print(f"Error during RandomForestClassifier training: {e}")
            logging.error(f"Error during RandomForestClassifier training: {e}")
            sys.exit(1)

        # Generate meta-features for Classification
        fold_meta_features_clf, meta_feature_names_clf = generate_meta_features_clf(
            fcnn2=fcnn2_clf,
            lgbm_clf=lgbm_clf,
            xgb_clf=xgb_clf,
            rf_clf=rf_clf,
            test_loader=test_loader_clf,
            X_test_scaled=X_test_clf_scaled,
            device=device
        )

        # Append to meta-feature lists for Classification
        meta_train_features_clf.append(fold_meta_features_clf)
        meta_train_labels_clf.extend(y_test_clf_fold.values)

        fold += 1

    # Combine all meta-features and labels for Regression
    meta_train_features_reg = np.vstack(meta_train_features_reg)
    meta_train_labels_reg = np.array(meta_train_labels_reg)

    # Combine all meta-features and labels for Classification
    meta_train_features_clf = np.vstack(meta_train_features_clf)
    meta_train_labels_clf = np.array(meta_train_labels_clf)

    # Train Meta-Model for Regression
    print("\nTraining Meta-Model for Regression...")
    logging.info("Training Meta-Model for Regression started.")
    try:
        meta_model_reg = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.15,
            max_depth=5,
            random_state=42
        )
        meta_model_reg.fit(meta_train_features_reg, meta_train_labels_reg)
        joblib.dump(meta_model_reg, 'meta_model_reg.joblib')
        logging.info("Meta-Model for Regression trained successfully.")
    except Exception as e:
        print(f"Error during Meta-Model Regression training: {e}")
        logging.error(f"Error during Meta-Model Regression training: {e}")
        sys.exit(1)

    # Train Meta-Model for Classification
    print("\nTraining Meta-Model for Classification...")
    logging.info("Training Meta-Model for Classification started.")
    try:
        meta_model_clf = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.15,
            max_depth=5,
            random_state=42
        )
        meta_model_clf.fit(meta_train_features_clf, meta_train_labels_clf)
        joblib.dump(meta_model_clf, 'meta_model_clf.joblib')
        logging.info("Meta-Model for Classification trained successfully.")
    except Exception as e:
        print(f"Error during Meta-Model Classification training: {e}")
        logging.error(f"Error during Meta-Model Classification training: {e}")
        sys.exit(1)

    # Feature importance analysis for Regression
    print("\nPerforming Feature Importance Analysis for Regression...")
    logging.info("Feature Importance Analysis for Regression started.")
    try:
        feature_importance_reg = feature_ablation_regression(
            meta_model_reg=meta_model_reg,
            X_test_meta=meta_train_features_reg,
            y_test_reg=meta_train_labels_reg,
            feature_names=meta_feature_names_reg
        )
        plot_feature_importance(feature_importance_reg, task='regression')
    except Exception as e:
        print(f"Error during Feature Importance Analysis for Regression: {e}")
        logging.error(f"Error during Feature Importance Analysis for Regression: {e}")
        sys.exit(1)

    # Feature importance analysis for Classification
    print("\nPerforming Feature Importance Analysis for Classification...")
    logging.info("Feature Importance Analysis for Classification started.")
    try:
        feature_importance_clf = feature_ablation_classification(
            meta_model_clf=meta_model_clf,
            X_test_meta=meta_train_features_clf,
            y_test_clf=meta_train_labels_clf,
            feature_names=meta_feature_names_clf
        )
        plot_feature_importance(feature_importance_clf, task='classification')
    except Exception as e:
        print(f"Error during Feature Importance Analysis for Classification: {e}")
        logging.error(f"Error during Feature Importance Analysis for Classification: {e}")
        sys.exit(1)

    # Execute Feature Ablation Analysis on Original Features
    feature_ablation_analysis(
        X=X,
        y_class=y_class,
        y_reg=y_reg,
        feature_list=original_features,
        device=device,
        scaler_type='StandardScaler'  # or 'RobustScaler' based on your preference
    )

    # Generate Feature Ablation Plots
    plot_feature_importance_ablation()

    print("\nFeature Ablation Analysis completed successfully.")
    print("Feature importance results saved to 'feature_ablation_importance.csv' and plots saved as PNG files.")


if __name__ == "__main__":
    main()
