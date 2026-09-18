import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from regression_scale_utils import regression_metrics_original_price
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
import ta  # Ensure ta library is installed: pip install ta
import warnings
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import autocast, GradScaler
import optuna

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


# Define the Dataset class for Regression
class FinancialRegressionDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets.astype(np.float32)  # Ensure targets are float for regression

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
        self.labels = labels.astype(np.int64)  # Ensure labels are int64

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
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
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

    # Check if all features are present
    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        raise ValueError(f"The following required features are missing from the data: {missing_features}")

    # Split features and targets
    X = data[features].values
    y_class = data['Next_Day_Change'].values
    y_reg = data['Next_Day_Price'].values

    return X, y_class, y_reg


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
    return X_train_scaled, X_test_scaled, scaler


# LightGBM objective for Regression
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
        'random_state': 42,
        'n_jobs': -1
    }

    lgbm = LGBMClassifier(**param)
    score = cross_val_score(lgbm, X_train, y_train, cv=3, scoring='roc_auc').mean()
    return score  # We want to maximize ROC AUC


# Optimize LightGBM for Classification
def optimize_lgbm_clf(X_train, y_train):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_lgbm_clf(trial, X_train, y_train), n_trials=50)

    print("Best parameters for LGBMClassifier:", study.best_params)
    print("Best ROC AUC for LGBMClassifier:", study.best_value)
    logging.info(f"Best parameters for LGBMClassifier: {study.best_params}")
    logging.info(f"Best ROC AUC for LGBMClassifier: {study.best_value}")

    return study.best_params


# Train base models for Regression
def train_base_models_reg(X_train, y_train, device):
    # Initialize FCNN Regressor
    hidden_dims = [256, 128, 64]
    fcnn = FCNNRegressor(input_dim=X_train.shape[1], hidden_dims=hidden_dims).to(device)

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
    optimizer = optim.AdamW(fcnn.parameters(), lr=0.005, weight_decay=1e-5)

    # Initialize learning rate scheduler with Cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=12)

    # Initialize MSE Loss
    mse_loss = nn.MSELoss()

    # Group models and related components
    models = (fcnn,)
    ensemble_models = (lgbm_reg, xgb_reg, rf_reg)
    optimizers = (optimizer,)
    schedulers = (scheduler,)
    early_stoppings = (early_stopping,)

    return models, ensemble_models, optimizers, schedulers, early_stoppings, mse_loss


# Train base models for Classification
def train_base_models_clf(X_train, y_train, device):
    # Initialize FCNN Classifier
    hidden_dims = [256, 128, 64]
    fcnn = FCNNClassifier(input_dim=X_train.shape[1], hidden_dims=hidden_dims).to(device)

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
    optimizer = optim.AdamW(fcnn.parameters(), lr=0.0005, weight_decay=1e-5)

    # Initialize learning rate scheduler with Cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=15)

    # Compute class weights and initialize Focal Loss
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    focal_loss = FocalLoss(alpha=class_weights, gamma=2)

    # Group models and related components
    models = (fcnn,)
    ensemble_models = (lgbm_clf, xgb_clf, rf_clf)
    optimizers = (optimizer,)
    schedulers = (scheduler,)
    early_stoppings = (early_stopping,)

    return models, ensemble_models, optimizers, schedulers, early_stoppings, focal_loss


# Train FCNN Regressor
def train_fcnn_reg(models, optimizers, schedulers, early_stoppings, mse_loss, train_loader, device, num_epochs=300):
    fcnn, = models  # single model
    optimizer, = optimizers
    scheduler, = schedulers
    early_stopping, = early_stoppings

    loss_history = []
    scaler = GradScaler()

    for epoch in range(num_epochs):
        fcnn.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Train FCNN
            optimizer.zero_grad()
            with autocast():
                outputs = fcnn(X_batch).squeeze()
                loss = mse_loss(outputs, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        # Calculate average loss
        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)

        # Print and log loss
        print(f"Epoch {epoch + 1} - FCNN Loss: {avg_loss:.4f}")
        logging.info(f"Epoch {epoch + 1} - FCNN Loss: {avg_loss:.4f}")

        # Step the scheduler
        scheduler.step()

        # Check early stopping
        early_stopping(avg_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            logging.info("Early stopping triggered.")
            break

    return loss_history,


# Train FCNN Classifier
def train_fcnn_clf(models, optimizers, schedulers, early_stoppings, focal_loss, train_loader, device, num_epochs=300):
    fcnn, = models  # single model
    optimizer, = optimizers
    scheduler, = schedulers
    early_stopping, = early_stoppings

    loss_history = []
    scaler = GradScaler()

    for epoch in range(num_epochs):
        fcnn.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Train FCNN
            optimizer.zero_grad()
            with autocast():
                outputs = fcnn(X_batch)
                loss = focal_loss(outputs, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        # Calculate average loss
        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)

        # Print and log loss
        print(f"Epoch {epoch + 1} - FCNN Loss: {avg_loss:.4f}")
        logging.info(f"Epoch {epoch + 1} - FCNN Loss: {avg_loss:.4f}")

        # Step the scheduler
        scheduler.step()

        # Check early stopping
        early_stopping(avg_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            logging.info("Early stopping triggered.")
            break

    return loss_history,


# Plot loss curves
def plot_loss_curve(loss_history, fold, task='regression', eval=False):
    plt.figure(figsize=(10, 6))
    if task in ['regression', 'classification']:
        plt.plot(loss_history, label='FCNN Loss', color='blue', linestyle='-')
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


# Define Normalized RMSE
def normalized_root_mean_squared_error(y_true, y_pred, y_train):
    range_y = y_train.max() - y_train.min()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse / range_y


# Generate meta-features for Regression
def generate_meta_features_reg(fcnn, lgbm_reg, xgb_reg, rf_reg, test_loader, X_test_scaled, device):
    fcnn.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            output = fcnn(X_batch).squeeze().cpu().numpy()
            preds.extend(output)

    # LightGBM predictions
    lgbm_preds = lgbm_reg.predict(X_test_scaled)

    # XGBoost predictions
    xgb_preds = xgb_reg.predict(X_test_scaled)

    # Random Forest predictions
    rf_preds = rf_reg.predict(X_test_scaled)

    # Stack all predictions as meta-features
    fold_meta_features = np.column_stack([
        preds,         # FCNN predictions (scaled)
        lgbm_preds,    # LightGBM predictions (scaled)
        xgb_preds,     # XGBoost predictions (scaled)
        rf_preds       # Random Forest predictions (scaled)
    ])
    return fold_meta_features


# Generate meta-features for Classification
def generate_meta_features_clf(fcnn, lgbm_clf, xgb_clf, rf_clf, test_loader, X_test_scaled, device):
    fcnn.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            output = fcnn(X_batch)
            preds.extend(torch.softmax(output, dim=1).cpu().numpy())  # Convert to probabilities

    # LightGBM predictions
    lgbm_probs = lgbm_clf.predict_proba(X_test_scaled)

    # XGBoost predictions
    xgb_probs = xgb_clf.predict_proba(X_test_scaled)

    # Random Forest predictions
    rf_probs = rf_clf.predict_proba(X_test_scaled)

    # Stack all predictions as meta-features
    fold_meta_features = np.hstack([
        np.array(preds),
        lgbm_probs,
        xgb_probs,
        rf_probs
    ])
    return fold_meta_features


# Evaluate Ensemble Models
def evaluate_ensemble_models(meta_model_reg, meta_train_features_reg, y_test_reg_fold):
    # Predict with Meta-Model
    fold_meta_preds_reg = meta_model_reg.predict(meta_train_features_reg)

    # Calculate RMSE on normalized data
    rmse = root_mean_squared_error(y_test_reg_fold, fold_meta_preds_reg)

    # Print and log RMSE
    print(f"Fold RMSE (on normalized data): {rmse:.4f}")
    logging.info(f"Fold RMSE (on normalized data): {rmse:.4f}")

    return rmse


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
    unique, counts = np.unique(y_class, return_counts=True)
    print(pd.Series(counts, index=unique, name='count'))

    logging.info(f"Classification Target Distribution:\n{pd.Series(counts, index=unique, name='count')}")

    print("\nRegression Target Statistics:")
    print(pd.Series(y_reg).describe())
    logging.info(f"Regression Target Statistics:\n{pd.Series(y_reg).describe()}")

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
        X_train_reg, X_test_reg = X[train_index], X[test_index]
        y_train_reg_fold, y_test_reg_fold = y_reg[train_index], y_reg[test_index]

        # Split data for Classification
        X_train_clf, X_test_clf = X[train_index], X[test_index]
        y_train_clf_fold, y_test_clf_fold = y_class[train_index], y_class[test_index]

        # Scale features for Regression
        X_train_reg_scaled, X_test_reg_scaled, scaler_X_reg = preprocess_and_scale(
            X_train_reg, X_test_reg, scaler_type='StandardScaler'
        )

        # Scale features for Classification
        X_train_clf_scaled, X_test_clf_scaled, scaler_X_clf = preprocess_and_scale(
            X_train_clf, X_test_clf, scaler_type='StandardScaler'
        )

        # Scale targets for Regression
        scaler_y_reg = StandardScaler()
        y_train_reg_scaled = scaler_y_reg.fit_transform(y_train_reg_fold.reshape(-1, 1)).flatten()
        y_test_reg_scaled = scaler_y_reg.transform(y_test_reg_fold.reshape(-1, 1)).flatten()

        # Create Datasets and DataLoaders for Regression
        train_dataset_reg = FinancialRegressionDataset(
            X_train_reg_scaled, y_train_reg_scaled)
        test_dataset_reg = FinancialRegressionDataset(
            X_test_reg_scaled, y_test_reg_scaled)
        train_loader_reg = DataLoader(train_dataset_reg, batch_size=32, shuffle=True)
        test_loader_reg = DataLoader(test_dataset_reg, batch_size=32, shuffle=False)

        # Create Datasets and DataLoaders for Classification
        train_dataset_clf = FinancialClassificationDataset(X_train_clf_scaled, y_train_clf_fold)
        test_dataset_clf = FinancialClassificationDataset(X_test_clf_scaled, y_test_clf_fold)
        train_loader_clf = DataLoader(train_dataset_clf, batch_size=32, shuffle=True)
        test_loader_clf = DataLoader(test_dataset_clf, batch_size=32, shuffle=False)

        # Train base models for Regression
        models_reg, ensemble_models_reg, optimizers_reg, schedulers_reg, early_stoppings_reg, mse_loss = train_base_models_reg(
            X_train_reg_scaled, y_train_reg_scaled, device
        )
        loss_history_reg, = train_fcnn_reg(
            models=models_reg,
            optimizers=optimizers_reg,
            schedulers=schedulers_reg,
            early_stoppings=early_stoppings_reg,
            mse_loss=mse_loss,
            train_loader=train_loader_reg,
            device=device
        )

        # Plot loss curves for Regression
        plot_loss_curve(loss_history_reg, fold, task='regression')

        # Get Regression Models
        fcnn_reg, = models_reg
        lgbm_reg, xgb_reg, rf_reg = ensemble_models_reg

        # Train LightGBM Regressor
        try:
            lgbm_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during LGBMRegressor training: {e}")
            logging.error(f"Error during LGBMRegressor training: {e}")
            return

        # Train XGBoost Regressor
        try:
            xgb_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during XGBoost Regressor training: {e}")
            logging.error(f"Error during XGBoost Regressor training: {e}")
            return

        # Train Random Forest Regressor
        try:
            rf_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during RandomForestRegressor training: {e}")
            logging.error(f"Error during RandomForestRegressor training: {e}")
            return

        # Generate meta-features for Regression
        fold_meta_features_reg = generate_meta_features_reg(
            fcnn=fcnn_reg,
            lgbm_reg=lgbm_reg,
            xgb_reg=xgb_reg,
            rf_reg=rf_reg,
            test_loader=test_loader_reg,
            X_test_scaled=X_test_reg_scaled,
            device=device
        )

        # Append to meta-feature lists for Regression
        meta_train_features_reg.append(fold_meta_features_reg)
        meta_train_labels_reg.extend(y_test_reg_scaled)

        # Train base models for Classification
        models_clf, ensemble_models_clf, optimizers_clf, schedulers_clf, early_stoppings_clf, focal_loss = train_base_models_clf(
            X_train_clf_scaled, y_train_clf_fold, device
        )
        loss_history_clf, = train_fcnn_clf(
            models=models_clf,
            optimizers=optimizers_clf,
            schedulers=schedulers_clf,
            early_stoppings=early_stoppings_clf,
            focal_loss=focal_loss,
            train_loader=train_loader_clf,
            device=device
        )

        # Plot loss curves for Classification
        plot_loss_curve(loss_history_clf, fold, task='classification')

        # Get Classification Models
        fcnn_clf, = models_clf
        lgbm_clf, xgb_clf, rf_clf = ensemble_models_clf

        # Train LightGBM Classifier
        try:
            lgbm_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during LGBMClassifier training: {e}")
            logging.error(f"Error during LGBMClassifier training: {e}")
            return

        # Train XGBoost Classifier
        try:
            xgb_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during XGBoost Classifier training: {e}")
            logging.error(f"Error during XGBoost Classifier training: {e}")
            return

        # Train Random Forest Classifier
        try:
            rf_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during RandomForestClassifier training: {e}")
            logging.error(f"Error during RandomForestClassifier training: {e}")
            return

        # Generate meta-features for Classification
        fold_meta_features_clf = generate_meta_features_clf(
            fcnn=fcnn_clf,
            lgbm_clf=lgbm_clf,
            xgb_clf=xgb_clf,
            rf_clf=rf_clf,
            test_loader=test_loader_clf,
            X_test_scaled=X_test_clf_scaled,
            device=device
        )

        # Append to meta-feature lists for Classification
        meta_train_features_clf.append(fold_meta_features_clf)
        meta_train_labels_clf.extend(y_test_clf_fold)

    # Combine all meta-features and labels for Regression
    meta_train_features_reg = np.vstack(meta_train_features_reg)
    meta_train_labels_reg = np.array(meta_train_labels_reg)

    # Combine all meta-features and labels for Classification
    meta_train_features_clf = np.vstack(meta_train_features_clf)
    meta_train_labels_clf = np.array(meta_train_labels_clf)

    # Train Meta-Model for Regression
    print("\nTraining Meta-Model for Regression...")
    logging.info("Training Meta-Model for Regression started.")
    meta_model_reg = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.15,
        max_depth=5,
        random_state=42
    )
    meta_model_reg.fit(meta_train_features_reg, meta_train_labels_reg)

    # Train Meta-Model for Classification
    print("\nTraining Meta-Model for Classification...")
    logging.info("Training Meta-Model for Classification started.")
    meta_model_clf = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.15,
        max_depth=5,
        random_state=42
    )
    meta_model_clf.fit(meta_train_features_clf, meta_train_labels_clf)

    # Second Cross-Validation: Evaluate Ensemble Models
    print("\nEvaluating Ensemble Models for Regression and Classification...")
    logging.info("Evaluating Ensemble Models started.")

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\nFold {fold + 1} - Evaluating Ensemble Models")
        logging.info(f"Fold {fold + 1} - Evaluating Ensemble Models started")

        # Split data for Regression
        X_train_reg, X_test_reg = X[train_index], X[test_index]
        y_train_reg_fold, y_test_reg_fold = y_reg[train_index], y_reg[test_index]

        # Split data for Classification
        X_train_clf, X_test_clf = X[train_index], X[test_index]
        y_train_clf_fold, y_test_clf_fold = y_class[train_index], y_class[test_index]

        # Scale features for Regression
        X_train_reg_scaled, X_test_reg_scaled, scaler_X_reg = preprocess_and_scale(
            X_train_reg, X_test_reg, scaler_type='StandardScaler'
        )

        # Scale features for Classification
        X_train_clf_scaled, X_test_clf_scaled, scaler_X_clf = preprocess_and_scale(
            X_train_clf, X_test_clf, scaler_type='StandardScaler'
        )

        # Scale targets for Regression
        scaler_y_reg = StandardScaler()
        y_train_reg_scaled = scaler_y_reg.fit_transform(y_train_reg_fold.reshape(-1, 1)).flatten()
        y_test_reg_scaled = scaler_y_reg.transform(y_test_reg_fold.reshape(-1, 1)).flatten()

        # Create Datasets and DataLoaders for Regression
        train_dataset_reg = FinancialRegressionDataset(
            X_train_reg_scaled, y_train_reg_scaled)
        test_dataset_reg = FinancialRegressionDataset(
            X_test_reg_scaled, y_test_reg_scaled)
        train_loader_reg = DataLoader(train_dataset_reg, batch_size=32, shuffle=True)
        test_loader_reg = DataLoader(test_dataset_reg, batch_size=32, shuffle=False)

        # Create Datasets and DataLoaders for Classification
        train_dataset_clf = FinancialClassificationDataset(X_train_clf_scaled, y_train_clf_fold)
        test_dataset_clf = FinancialClassificationDataset(X_test_clf_scaled, y_test_clf_fold)
        train_loader_clf = DataLoader(train_dataset_clf, batch_size=32, shuffle=True)
        test_loader_clf = DataLoader(test_dataset_clf, batch_size=32, shuffle=False)

        # Train base models for Regression
        models_reg, ensemble_models_reg, optimizers_reg, schedulers_reg, early_stoppings_reg, mse_loss = train_base_models_reg(
            X_train_reg_scaled, y_train_reg_scaled, device
        )
        loss_history_reg, = train_fcnn_reg(
            models=models_reg,
            optimizers=optimizers_reg,
            schedulers=schedulers_reg,
            early_stoppings=early_stoppings_reg,
            mse_loss=mse_loss,
            train_loader=train_loader_reg,
            device=device
        )

        # Plot loss curves for Regression
        plot_loss_curve(loss_history_reg, fold, task='regression', eval=True)

        # Get Regression Models
        fcnn_reg, = models_reg
        lgbm_reg, xgb_reg, rf_reg = ensemble_models_reg

        # Train LightGBM Regressor
        try:
            lgbm_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during LGBMRegressor training: {e}")
            logging.error(f"Error during LGBMRegressor training: {e}")
            return

        # Train XGBoost Regressor
        try:
            xgb_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during XGBoost Regressor training: {e}")
            logging.error(f"Error during XGBoost Regressor training: {e}")
            return

        # Train Random Forest Regressor
        try:
            rf_reg.fit(X_train_reg_scaled, y_train_reg_scaled)
        except Exception as e:
            print(f"Error during RandomForestRegressor training: {e}")
            logging.error(f"Error during RandomForestRegressor training: {e}")
            return

        # Generate meta-features for Regression
        fold_meta_features_reg = generate_meta_features_reg(
            fcnn=fcnn_reg,
            lgbm_reg=lgbm_reg,
            xgb_reg=xgb_reg,
            rf_reg=rf_reg,
            test_loader=test_loader_reg,
            X_test_scaled=X_test_reg_scaled,
            device=device
        )

        # Predict with Meta-Model for Regression
        fold_meta_preds_reg = meta_model_reg.predict(fold_meta_features_reg)

        # Calculate Regression metrics (RMSE on normalized data)
        fold_meta_preds_reg, rmse, mape = regression_metrics_original_price(
            y_test_reg_fold, fold_meta_preds_reg, prediction_scaler=scaler_y_reg
        )

        # Store Regression metrics
        rmses.append(rmse)
        mapes.append(mape)

        # Print and log Regression metrics
        print(f"Fold {fold + 1} - Regression: RMSE (USD/oz)={rmse:.4f}, MAPE (%)={mape:.4f}")
        logging.info(f"Fold {fold + 1} - Regression: RMSE (USD/oz)={rmse:.4f}, MAPE (%)={mape:.4f}")

        # Train base models for Classification
        models_clf, ensemble_models_clf, optimizers_clf, schedulers_clf, early_stoppings_clf, focal_loss = train_base_models_clf(
            X_train_clf_scaled, y_train_clf_fold, device
        )
        loss_history_clf, = train_fcnn_clf(
            models=models_clf,
            optimizers=optimizers_clf,
            schedulers=schedulers_clf,
            early_stoppings=early_stoppings_clf,
            focal_loss=focal_loss,
            train_loader=train_loader_clf,
            device=device
        )

        # Plot loss curves for Classification
        plot_loss_curve(loss_history_clf, fold, task='classification', eval=True)

        # Get Classification Models
        fcnn_clf, = models_clf
        lgbm_clf, xgb_clf, rf_clf = ensemble_models_clf

        # Train LightGBM Classifier
        try:
            lgbm_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during LGBMClassifier training: {e}")
            logging.error(f"Error during LGBMClassifier training: {e}")
            return

        # Train XGBoost Classifier
        try:
            xgb_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during XGBoost Classifier training: {e}")
            logging.error(f"Error during XGBoost Classifier training: {e}")
            return

        # Train Random Forest Classifier
        try:
            rf_clf.fit(X_train_clf_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during RandomForestClassifier training: {e}")
            logging.error(f"Error during RandomForestClassifier training: {e}")
            return

        # Generate meta-features for Classification
        fold_meta_features_clf = generate_meta_features_clf(
            fcnn=fcnn_clf,
            lgbm_clf=lgbm_clf,
            xgb_clf=xgb_clf,
            rf_clf=rf_clf,
            test_loader=test_loader_clf,
            X_test_scaled=X_test_clf_scaled,
            device=device
        )

        # Predict with Meta-Model for Classification
        fold_meta_preds_clf = meta_model_clf.predict(fold_meta_features_clf)
        fold_meta_probs_clf = meta_model_clf.predict_proba(fold_meta_features_clf)[:, 1]

        # Calculate Classification metrics
        acc = accuracy_score(y_test_clf_fold, fold_meta_preds_clf)
        prec = precision_score(y_test_clf_fold, fold_meta_preds_clf)
        rec = recall_score(y_test_clf_fold, fold_meta_preds_clf)
        f1 = f1_score(y_test_clf_fold, fold_meta_preds_clf)
        auc = roc_auc_score(y_test_clf_fold, fold_meta_probs_clf)

        # Store Classification metrics
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        aucs.append(auc)

        # Print and log Classification metrics
        print(
            f"Fold {fold + 1} - Classification: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
        logging.info(
            f"Fold {fold + 1} - Classification: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    # Overall Results for Regression
    print("\nOverall Regression Results:")
    print(f"Average RMSE (USD/oz): {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"Average MAPE (%): {np.mean(mapes):.4f} ± {np.std(mapes):.4f}")
    logging.info(
        f"Overall Regression Results: "
        f"Average RMSE (USD/oz)={np.mean(rmses):.4f} ± {np.std(rmses):.4f}, "
        f"Average MAPE (%)={np.mean(mapes):.4f} ± {np.std(mapes):.4f}"
    )

    # Overall Results for Classification
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

    # Save Overall Results
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
