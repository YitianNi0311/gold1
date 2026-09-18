import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from torch.utils.data import Dataset, DataLoader
import logging
import matplotlib.pyplot as plt
import ta  # Ensure ta is installed: pip install ta
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


# Define the Dataset class
class FinancialDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels.values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


# Define Focal Loss
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


# Define the Fully Connected Neural Network Model
class FCNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5):
        super(FCNNModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU())  # Use LeakyReLU
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)  # Raw logits


def preprocess_data(file_path):
    # 加载数据
    data = pd.read_excel(file_path)

    # 确保 'Date' 列为 datetime 类型
    if not np.issubdtype(data['Date'].dtype, np.datetime64):
        data['Date'] = pd.to_datetime(data['Date'])

    # 创建目标变量
    data['Next_Day_Change'] = (data['GOLD'].diff() > 0).astype(int)

    # 特征工程
    # 滞后特征（历史数据，不需要下移）
    for lag in range(1, 31):
        data[f'GOLD_lag{lag}'] = data['GOLD'].shift(lag)

    # 移动平均（历史数据，不需要下移）
    data['GOLD_MA_3_days'] = data['GOLD'].rolling(window=3).mean()
    data['GOLD_MA_5_days'] = data['GOLD'].rolling(window=5).mean()
    data['GOLD_MA_10_days'] = data['GOLD'].rolling(window=10).mean()

    # 技术指标（历史数据，不需要下移）
    data['RSI'] = ta.momentum.RSIIndicator(close=data['GOLD'], window=14).rsi()
    macd = ta.trend.MACD(close=data['GOLD'])
    data['MACD'] = macd.macd_diff()

    # 布林带（历史数据，不需要下移）
    bollinger = ta.volatility.BollingerBands(close=data['GOLD'], window=20)
    data['Bollinger_High'] = bollinger.bollinger_hband()
    data['Bollinger_Low'] = bollinger.bollinger_lband()

    # 滚动统计特征（历史数据，不需要下移）
    rolling_window = 10
    data['GOLD_roll_mean_5'] = data['GOLD'].rolling(window=rolling_window).mean()
    data['GOLD_roll_std_5'] = data['GOLD'].rolling(window=rolling_window).std()
    data['GOLD_roll_min_5'] = data['GOLD'].rolling(window=rolling_window).min()
    data['GOLD_roll_max_5'] = data['GOLD'].rolling(window=rolling_window).max()

    # 自定义特征（当天数据，下移一行）
    data['Gold_Oil_Ratio'] = data['GOLD'] / data['CrudeOil_SpotPrice_BrentUK']
    data['USD_CNY_to_JPY'] = data['SpotRate_USD_CNY'] / data['SpotRate_Tokyo_9AM_USD_JPY']
    data['China_US_CPI_Ratio'] = data['China_CPI_YoY_CurrentMonth'] / data['US_CPI_YoY_NSA']

    # 变化率特征（当天数据，下移一行）
    data['Gold_Rate_of_Change'] = data['GOLD'].pct_change(periods=5)
    data['Oil_Rate_of_Change'] = data['CrudeOil_SpotPrice_BrentUK'].pct_change(periods=5)

    # 趋势特征（当天数据，下移一行）
    data['Gold_Trend_7_days'] = data['GOLD'].rolling(window=7).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )

    # 通胀调整金价（当天数据，下移一行）
    data['Gold_Inflation_Adjusted'] = data['GOLD'] / data['US_CPI_YoY_NSA']

    # 波动性特征（当天数据，下移一行）
    data['Gold_Volatility_10_days'] = data['GOLD'].rolling(window=10).std()
    data['Oil_Volatility_10_days'] = data['CrudeOil_SpotPrice_BrentUK'].rolling(window=10).std()

    # 交互特征（当天数据，下移一行）
    data['Gold_SP500_Ratio'] = data['GOLD'] / data['US_SP500_Index']
    data['US_Japan_Interest_Rate_Diff'] = data['US_DowJones_IndustrialAverage'] - data['SpotRate_Tokyo_9AM_USD_JPY']

    # 时间相关特征
    data['Month'] = data['Date'].dt.month
    data['Quarter'] = data['Date'].dt.quarter
    data['Day_of_Week'] = data['Date'].dt.dayofweek

    # 下移当天数据特征
    today_features = [
        'Gold_Oil_Ratio', 'USD_CNY_to_JPY', 'China_US_CPI_Ratio',
        'Gold_Rate_of_Change', 'Oil_Rate_of_Change', 'Gold_Trend_7_days',
        'Gold_Inflation_Adjusted', 'Gold_Volatility_10_days', 'Oil_Volatility_10_days',
        'Gold_SP500_Ratio', 'US_Japan_Interest_Rate_Diff'
    ]
    data[today_features] = data[today_features].shift(1)

    # 清理数据：删除 NaN 或 inf 值
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    # 定义特征列表
    features = [
        'GOLD_MA_3_days', 'GOLD_lag1', 'GOLD_MA_5_days',
        'GOLD_lag2', 'GOLD_MA_10_days', 'GOLD_lag3', 'GOLD_lag4', 'GOLD_lag5','GOLD_lag6','GOLD_lag7','GOLD_lag8','GOLD_lag9','GOLD_lag10',
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
        'Interbank_OpenRate_USD_INR', 'SpotRate_Tokyo_9AM_USD_JPY',
    ]

    # 检查特征是否齐全
    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        raise ValueError(f"The following required features are missing from the data: {missing_features}")

    # 划分特征和目标变量
    X = data[features]
    y = data['Next_Day_Change']

    return X, y


def preprocess_and_scale(X_train, X_test):
    # 使用RobustScaler以减少异常值影响
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def objective_lgbm(trial, X_train, y_train):
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


def optimize_lgbm(X_train, y_train):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_lgbm(trial, X_train, y_train), n_trials=50)

    print("Best parameters:", study.best_params)
    print("Best ROC AUC:", study.best_value)

    return study.best_params


def train_base_models(X_train, y_train, device):
    # Initialize FCNN1
    hidden_dims1 = [256, 128, 64]
    fcnn1 = FCNNModel(input_dim=X_train.shape[1], hidden_dims=hidden_dims1, output_dim=2, dropout=0.5).to(device)

    # Initialize FCNN2
    hidden_dims2 = [512, 256, 128, 64]
    fcnn2 = FCNNModel(input_dim=X_train.shape[1], hidden_dims=hidden_dims2, output_dim=2, dropout=0.5).to(device)

    # Optimize LightGBM
    best_params_lgbm = optimize_lgbm(X_train, y_train)
    lgbm_model = LGBMClassifier(
        **best_params_lgbm,
        random_state=42,
        n_jobs=-1
    )

    # Initialize XGBoost with improved parameters
    xgb_model = XGBClassifier(
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

    # Initialize Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    # Initialize optimizers with AdamW
    optimizer1 = optim.AdamW(fcnn1.parameters(), lr=0.0005, weight_decay=1e-63)
    optimizer2 = optim.AdamW(fcnn2.parameters(), lr=0.0005, weight_decay=1e-63)

    # Initialize learning rate schedulers with Cosine Annealing
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=50)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=50)

    # Initialize Early Stopping
    early_stopping1 = EarlyStopping(patience=15)
    early_stopping2 = EarlyStopping(patience=15)

    # Initialize Focal Loss with class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    focal_loss = FocalLoss(alpha=class_weights, gamma=2)

    # Group all models and related components
    models = (fcnn1, fcnn2, lgbm_model, xgb_model, rf_model)
    optimizers = (optimizer1, optimizer2)
    schedulers = (scheduler1, scheduler2)
    early_stoppings = (early_stopping1, early_stopping2)

    return models, optimizers, schedulers, early_stoppings, focal_loss


def train_fcnn(models, optimizers, schedulers, early_stoppings, focal_loss, train_loader, device, num_epochs=300):
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

            # Train FCNN1
            optimizer1.zero_grad()
            with autocast():
                outputs1 = fcnn1(X_batch)
                loss1 = focal_loss(outputs1, y_batch)
            scaler.scale(loss1).backward()
            scaler.step(optimizer1)
            scaler.update()
            running_loss1 += loss1.item()

            # Train FCNN2
            optimizer2.zero_grad()
            with autocast():
                outputs2 = fcnn2(X_batch)
                loss2 = focal_loss(outputs2, y_batch)
            scaler.scale(loss2).backward()
            scaler.step(optimizer2)
            scaler.update()
            running_loss2 += loss2.item()

        # Calculate average losses
        avg_loss1 = running_loss1 / len(train_loader)
        avg_loss2 = running_loss2 / len(train_loader)
        loss_history1.append(avg_loss1)
        loss_history2.append(avg_loss2)

        # Print and log losses
        print(f"Epoch {epoch + 1} - FCNN1 Loss: {avg_loss1:.4f}, FCNN2 Loss: {avg_loss2:.4f}")
        logging.info(f"Epoch {epoch + 1} - FCNN1 Loss: {avg_loss1:.4f}, FCNN2 Loss: {avg_loss2:.4f}")

        # Step the schedulers
        scheduler1.step()
        scheduler2.step()

        # Check early stopping
        early_stopping1(avg_loss1)
        early_stopping2(avg_loss2)
        if early_stopping1.early_stop or early_stopping2.early_stop:
            print("Early stopping triggered.")
            logging.info("Early stopping triggered.")
            break

    return loss_history1, loss_history2


def plot_loss_curve(loss_history1, loss_history2, fold, eval=False):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history1, label='FCNN1 Loss', color='blue', linestyle='-')
    plt.plot(loss_history2, label='FCNN2 Loss', color='orange', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    title = f'Training Loss Curve - Fold {fold + 1}'
    if eval:
        title += ' (Evaluation)'
    plt.title(title)
    plt.legend()
    plt.grid(True)
    filename = f'training_loss_curve_fold_{fold + 1}.png'
    if eval:
        filename = f'training_loss_curve_eval_fold_{fold + 1}.png'
    plt.savefig(filename)  # Save the plot as a file
    plt.close()  # Close the plot to avoid display issues


def generate_meta_features(fcnn1, fcnn2, lgbm_model, xgb_model, rf_model, test_loader, X_test_scaled, device):
    fcnn1.eval()
    fcnn2.eval()
    preds1, preds2 = [], []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            output1 = fcnn1(X_batch)
            output2 = fcnn2(X_batch)
            preds1.extend(torch.softmax(output1, dim=1).cpu().numpy())  # Convert to probabilities
            preds2.extend(torch.softmax(output2, dim=1).cpu().numpy())

    # LightGBM predictions
    lgbm_probs = lgbm_model.predict_proba(X_test_scaled)

    # XGBoost predictions
    xgb_probs = xgb_model.predict_proba(X_test_scaled)

    # Random Forest predictions
    rf_probs = rf_model.predict_proba(X_test_scaled)

    # Stack all predictions as meta-features
    fold_meta_features = np.hstack([
        np.array(preds1),
        np.array(preds2),
        lgbm_probs,
        xgb_probs,
        rf_probs
    ])
    return fold_meta_features


def main():
    # File path (modify as needed)
    file_path = "/home/w/桌面/lilj/GOLD_cleaned.xlsx"
    try:
        X, y = preprocess_data(file_path)
    except Exception as e:
        print(f"Error in preprocessing data: {e}")
        logging.error(f"Error in preprocessing data: {e}")
        return

    # Check class distribution
    print("Class Distribution:")
    print(y.value_counts())
    logging.info(f"Class Distribution:\n{y.value_counts()}")

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logging.info(f"Using device: {device}")

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    # Lists to store meta-features and labels
    meta_train_features = []
    meta_train_labels = []

    # Lists to store evaluation metrics
    accuracies, precisions, recalls, f1_scores, aucs = [], [], [], [], []

    # First Cross-Validation: Generate Meta-Features
    print("Generating meta-features and training base models...")
    logging.info("Generating meta-features and training base models started.")

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"Fold {fold + 1} - Training base models")
        logging.info(f"Fold {fold + 1} - Training base models started")

        # Split data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create Datasets and DataLoaders
        train_dataset = FinancialDataset(X_train_scaled, y_train)
        test_dataset = FinancialDataset(X_test_scaled, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Increased Batch Size
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Train base models
        models, optimizers, schedulers, early_stoppings, focal_loss = train_base_models(X_train_scaled, y_train, device)
        loss_history1, loss_history2 = train_fcnn(
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            early_stoppings=early_stoppings,
            focal_loss=focal_loss,
            train_loader=train_loader,
            device=device
        )

        # Plot loss curves
        plot_loss_curve(loss_history1, loss_history2, fold)

        # Get models
        fcnn1, fcnn2, lgbm_model, xgb_model, rf_model = models

        # Train LightGBM
        try:
            lgbm_model.fit(
                X_train_scaled, y_train
            )
        except Exception as e:
            print(f"Error during LightGBM training: {e}")
            logging.error(f"Error during LightGBM training: {e}")
            return

        # Train XGBoost
        try:
            xgb_model.fit(
                X_train_scaled, y_train
            )
        except Exception as e:
            print(f"Error during XGBoost training: {e}")
            logging.error(f"Error during XGBoost training: {e}")
            return

        # Train Random Forest
        try:
            rf_model.fit(
                X_train_scaled, y_train
            )
        except Exception as e:
            print(f"Error during Random Forest training: {e}")
            logging.error(f"Error during Random Forest training: {e}")
            return

        # Generate meta-features
        fold_meta_features = generate_meta_features(
            fcnn1=fcnn1,
            fcnn2=fcnn2,
            lgbm_model=lgbm_model,
            xgb_model=xgb_model,
            rf_model=rf_model,
            test_loader=test_loader,
            X_test_scaled=X_test_scaled,
            device=device
        )

        # Append to meta-feature lists
        meta_train_features.append(fold_meta_features)
        meta_train_labels.extend(y_test.values)

    # Combine all meta-features and labels
    meta_train_features = np.vstack(meta_train_features)
    meta_train_labels = np.array(meta_train_labels)

    # Train Meta-Model
    print("Training Meta-Model...")
    logging.info("Training Meta-Model started.")
    meta_model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.15, max_depth=5, random_state=42)
    meta_model.fit(meta_train_features, meta_train_labels)

    # Second Cross-Validation: Evaluate Ensemble Model
    print("Evaluating Ensemble Model...")
    logging.info("Evaluating Ensemble Model started.")

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"Fold {fold + 1} - Evaluating ensemble model")
        logging.info(f"Fold {fold + 1} - Evaluating ensemble model started")

        # Split data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create Datasets and DataLoaders
        train_dataset = FinancialDataset(X_train_scaled, y_train)
        test_dataset = FinancialDataset(X_test_scaled, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Train base models
        models, optimizers, schedulers, early_stoppings, focal_loss = train_base_models(X_train_scaled, y_train, device)
        loss_history1, loss_history2 = train_fcnn(
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            early_stoppings=early_stoppings,
            focal_loss=focal_loss,
            train_loader=train_loader,
            device=device
        )

        # Plot loss curves
        plot_loss_curve(loss_history1, loss_history2, fold, eval=True)

        # Get models
        fcnn1, fcnn2, lgbm_model, xgb_model, rf_model = models

        # Train LightGBM
        try:
            lgbm_model.fit(
                X_train_scaled, y_train
            )
        except Exception as e:
            print(f"Error during LightGBM training: {e}")
            logging.error(f"Error during LightGBM training: {e}")
            return

        # Train XGBoost
        try:
            xgb_model.fit(
                X_train_scaled, y_train
            )
        except Exception as e:
            print(f"Error during XGBoost training: {e}")
            logging.error(f"Error during XGBoost training: {e}")
            return

        # Train Random Forest
        try:
            rf_model.fit(
                X_train_scaled, y_train
            )
        except Exception as e:
            print(f"Error during Random Forest training: {e}")
            logging.error(f"Error during Random Forest training: {e}")
            return

        # Generate meta-features
        fold_meta_features = generate_meta_features(
            fcnn1=fcnn1,
            fcnn2=fcnn2,
            lgbm_model=lgbm_model,
            xgb_model=xgb_model,
            rf_model=rf_model,
            test_loader=test_loader,
            X_test_scaled=X_test_scaled,
            device=device
        )

        # Predict with Meta-Model
        fold_meta_preds = meta_model.predict(fold_meta_features)
        fold_meta_probs = meta_model.predict_proba(fold_meta_features)[:, 1]

        # Calculate evaluation metrics
        acc = accuracy_score(y_test, fold_meta_preds)
        prec = precision_score(y_test, fold_meta_preds)
        rec = recall_score(y_test, fold_meta_preds)
        f1 = f1_score(y_test, fold_meta_preds)
        auc = roc_auc_score(y_test, fold_meta_probs)

        # Store metrics
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        aucs.append(auc)

        # Print and log metrics
        print(
            f"Fold {fold + 1}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
        logging.info(
            f"Fold {fold + 1}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}"
        )

    # Overall Results
    print("\nOverall Results:")
    print(f"Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Average Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"Average Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"Average F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"Average AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    logging.info(
        f"Overall Results: "
        f"Average Accuracy={np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}, "
        f"Average Precision={np.mean(precisions):.4f} ± {np.std(precisions):.4f}, "
        f"Average Recall={np.mean(recalls):.4f} ± {np.std(recalls):.4f}, "
        f"Average F1 Score={np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}, "
        f"Average AUC={np.mean(aucs):.4f} ± {np.std(aucs):.4f}"
    )


if __name__ == "__main__":
    main()
