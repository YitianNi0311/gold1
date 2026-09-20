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
import ta  # 确保已安装 ta 库: pip install ta
import warnings
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import autocast, GradScaler
import optuna

warnings.filterwarnings('ignore')  # 抑制警告

# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 配置日志记录
logging.basicConfig(
    filename='training_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# 定义多任务数据集类
class FinancialMultiTaskDataset(Dataset):
    def __init__(self, features, labels_class, labels_reg):
        self.features = features
        self.labels_class = labels_class.values.astype(np.int64)
        self.labels_reg = labels_reg.values.astype(np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels_class[idx], dtype=torch.long),
            torch.tensor(self.labels_reg[idx], dtype=torch.float32)
        )


# 定义早停机制
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


# 定义焦点损失（Focal Loss）用于分类任务
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)  # 原始 logits
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


# 定义多任务全连接神经网络模型
class MultiTaskFCNN(nn.Module):
    def __init__(self, input_dim, shared_hidden_dims, task_hidden_dims, dropout=0.5):
        super(MultiTaskFCNN, self).__init__()

        # 构建共享网络
        shared_layers = []
        prev_dim = input_dim
        for hidden_dim in shared_hidden_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.BatchNorm1d(hidden_dim))
            shared_layers.append(nn.LeakyReLU())
            shared_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.shared_network = nn.Sequential(*shared_layers)

        # 构建回归头部
        reg_layers = []
        reg_prev_dim = prev_dim  # 共享网络的输出维度
        for hidden_dim in task_hidden_dims['reg']:
            reg_layers.append(nn.Linear(reg_prev_dim, hidden_dim))
            reg_layers.append(nn.BatchNorm1d(hidden_dim))
            reg_layers.append(nn.LeakyReLU())
            reg_layers.append(nn.Dropout(dropout))
            reg_prev_dim = hidden_dim
        reg_layers.append(nn.Linear(reg_prev_dim, 1))
        self.regression_head = nn.Sequential(*reg_layers)

        # 构建分类头部
        clf_layers = []
        clf_prev_dim = prev_dim  # 共享网络的输出维度
        for hidden_dim in task_hidden_dims['clf']:
            clf_layers.append(nn.Linear(clf_prev_dim, hidden_dim))
            clf_layers.append(nn.BatchNorm1d(hidden_dim))
            clf_layers.append(nn.LeakyReLU())
            clf_layers.append(nn.Dropout(dropout))
            clf_prev_dim = hidden_dim
        clf_layers.append(nn.Linear(clf_prev_dim, 2))  # 假设是二分类
        self.classification_head = nn.Sequential(*clf_layers)

    def forward(self, x):
        shared_representation = self.shared_network(x)
        reg_output = self.regression_head(shared_representation).squeeze(1)  # 输出形状: (batch_size,)
        clf_output = self.classification_head(shared_representation)  # 原始 logits
        return clf_output, reg_output


# 定义均方根误差（RMSE）
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# 定义自定义的平均绝对百分比误差（MAPE）
def mean_absolute_percentage_error_custom(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100.0


# 定义数据预处理函数
def preprocess_data(file_path):
    # 加载数据
    data = pd.read_excel(file_path)

    # 确保 'Date' 列是 datetime 类型
    if not np.issubdtype(data['Date'].dtype, np.datetime64):
        data['Date'] = pd.to_datetime(data['Date'])

    # 创建分类目标：如果下一个交易日的黄金价格上涨，则为 1，否则为 0
    data['Next_Day_Change'] = (data['GOLD'].diff() > 0).astype(int)

    # 创建回归目标：下一个交易日的黄金价格
    data['Next_Day_Price'] = data['GOLD'].shift(-1)

    # 特征工程
    # 滞后特征
    for lag in range(1, 31):
        data[f'GOLD_lag{lag}'] = data['GOLD'].shift(lag)

    # 移动平均
    data['GOLD_MA_3_days'] = data['GOLD'].rolling(window=3).mean()
    data['GOLD_MA_5_days'] = data['GOLD'].rolling(window=5).mean()
    data['GOLD_MA_10_days'] = data['GOLD'].rolling(window=10).mean()

    # 技术指标
    data['RSI'] = ta.momentum.RSIIndicator(close=data['GOLD'], window=14).rsi()
    macd = ta.trend.MACD(close=data['GOLD'])
    data['MACD'] = macd.macd_diff()

    # 布林带
    bollinger = ta.volatility.BollingerBands(close=data['GOLD'], window=20)
    data['Bollinger_High'] = bollinger.bollinger_hband()
    data['Bollinger_Low'] = bollinger.bollinger_lband()

    # 滚动统计
    rolling_window = 10
    data['GOLD_roll_mean_5'] = data['GOLD'].rolling(window=rolling_window).mean()
    data['GOLD_roll_std_5'] = data['GOLD'].rolling(window=rolling_window).std()
    data['GOLD_roll_min_5'] = data['GOLD'].rolling(window=rolling_window).min()
    data['GOLD_roll_max_5'] = data['GOLD'].rolling(window=rolling_window).max()

    # 自定义特征
    data['Gold_Oil_Ratio'] = data['GOLD'] / data['CrudeOil_SpotPrice_BrentUK']
    data['USD_CNY_to_JPY'] = data['SpotRate_USD_CNY'] / data['SpotRate_Tokyo_9AM_USD_JPY']
    data['China_US_CPI_Ratio'] = data['China_CPI_YoY_CurrentMonth'] / data['US_CPI_YoY_NSA']

    # 变化率特征
    data['Gold_Rate_of_Change'] = data['GOLD'].pct_change(periods=5)
    data['Oil_Rate_of_Change'] = data['CrudeOil_SpotPrice_BrentUK'].pct_change(periods=5)

    # 趋势特征
    data['Gold_Trend_7_days'] = data['GOLD'].rolling(window=7).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )

    # 通胀调整的黄金价格
    data['Gold_Inflation_Adjusted'] = data['GOLD'] / data['US_CPI_YoY_NSA']

    # 波动率特征
    data['Gold_Volatility_10_days'] = data['GOLD'].rolling(window=10).std()
    data['Oil_Volatility_10_days'] = data['CrudeOil_SpotPrice_BrentUK'].rolling(window=10).std()

    # 交互特征
    data['Gold_SP500_Ratio'] = data['GOLD'] / data['US_SP500_Index']
    data['US_Japan_Interest_Rate_Diff'] = data['US_DowJones_IndustrialAverage'] - data['SpotRate_Tokyo_9AM_USD_JPY']

    # 时间相关特征
    data['Month'] = data['Date'].dt.month
    data['Quarter'] = data['Date'].dt.quarter
    data['Day_of_Week'] = data['Date'].dt.dayofweek

    # 将今天的特征向后移动一天
    today_features = [
        'Gold_Oil_Ratio', 'USD_CNY_to_JPY', 'China_US_CPI_Ratio',
        'Gold_Rate_of_Change', 'Oil_Rate_of_Change', 'Gold_Trend_7_days',
        'Gold_Inflation_Adjusted', 'Gold_Volatility_10_days', 'Oil_Volatility_10_days',
        'Gold_SP500_Ratio', 'US_Japan_Interest_Rate_Diff'
    ]
    data[today_features] = data[today_features].shift(1)

    # 清理数据：移除 NaN 或 inf
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    # 定义特征列表
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

    # 检查是否所有特征都存在
    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        raise ValueError(f"The following required features are missing from the data: {missing_features}")

    # 分割特征和目标
    X = data[features]
    y_class = data['Next_Day_Change']
    y_reg = data['Next_Day_Price']

    return X, y_class, y_reg


# 定义特征缩放函数
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


# LightGBM 回归的目标函数
def objective_lgbm_reg(trial, X_train, y_train):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),  # 防止过拟合
        'random_state': 42,
        'n_jobs': -1,
        'metric': 'rmse',
        'verbose': -1
    }

    lgbm = LGBMRegressor(**param)
    score = cross_val_score(lgbm, X_train, y_train, cv=3, scoring='neg_mean_squared_error').mean()
    return -score  # 我们希望最小化 MSE


# LightGBM 分类的目标函数
def objective_lgbm_clf(trial, X_train, y_train):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),  # 防止过拟合
        'random_state': 42,
        'n_jobs': -1,
        'metric': 'auc',
        'verbose': -1,
        'class_weight': 'balanced'  # 处理类别不平衡
    }

    lgbm = LGBMClassifier(**param)
    score = cross_val_score(lgbm, X_train, y_train, cv=3, scoring='roc_auc').mean()
    return score  # 我们希望最大化 ROC AUC


# 优化 LightGBM 回归模型的超参数
def optimize_lgbm_reg(X_train, y_train):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_lgbm_reg(trial, X_train, y_train), n_trials=50)

    print("Best parameters for LGBMRegressor:", study.best_params)
    print("Best MSE for LGBMRegressor:", study.best_value)
    logging.info(f"Best parameters for LGBMRegressor: {study.best_params}")
    logging.info(f"Best MSE for LGBMRegressor: {study.best_value}")

    return study.best_params


# 优化 LightGBM 分类模型的超参数
def optimize_lgbm_clf(X_train, y_train):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_lgbm_clf(trial, X_train, y_train), n_trials=50)

    print("Best parameters for LGBMClassifier:", study.best_params)
    print("Best ROC AUC for LGBMClassifier:", study.best_value)
    logging.info(f"Best parameters for LGBMClassifier: {study.best_params}")
    logging.info(f"Best ROC AUC for LGBMClassifier: {study.best_value}")

    return study.best_params


# 训练多任务学习的基础模型
def train_base_models_mtl(X_train, y_class_train, y_reg_train, device):
    # 初始化多任务 FCNN 模型
    shared_hidden_dims = [256, 128]
    task_hidden_dims = {
        'reg': [64],
        'clf': [64]
    }
    mtl_model = MultiTaskFCNN(input_dim=X_train.shape[1],
                              shared_hidden_dims=shared_hidden_dims,
                              task_hidden_dims=task_hidden_dims,
                              dropout=0.5).to(device)

    # 优化并初始化 LightGBM 回归器
    best_params_lgbm_reg = optimize_lgbm_reg(X_train, y_reg_train)
    lgbm_reg = LGBMRegressor(**best_params_lgbm_reg, random_state=42, n_jobs=-1)

    # 优化并初始化 LightGBM 分类器
    best_params_lgbm_clf = optimize_lgbm_clf(X_train, y_class_train)
    lgbm_clf = LGBMClassifier(**best_params_lgbm_clf, random_state=42, n_jobs=-1)

    # 初始化 XGBoost 回归器
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

    # 初始化 XGBoost 分类器
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

    # 初始化随机森林回归器
    rf_reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    # 初始化随机森林分类器
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    # 初始化优化器
    optimizer = optim.AdamW(mtl_model.parameters(), lr=0.005, weight_decay=1e-5)

    # 初始化学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # 初始化早停机制
    early_stopping = EarlyStopping(patience=15)

    # 计算分类任务的类权重并初始化焦点损失
    class_weights = compute_class_weight('balanced', classes=np.unique(y_class_train), y=y_class_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    focal_loss = FocalLoss(alpha=class_weights, gamma=2)

    # 初始化损失函数
    mse_loss = nn.MSELoss()

    # 组合所有模型和相关组件
    models = (mtl_model, lgbm_reg, lgbm_clf, xgb_reg, xgb_clf, rf_reg, rf_clf)
    optimizers = (optimizer,)
    schedulers = (scheduler,)
    early_stoppings = (early_stopping,)
    losses = (mse_loss, focal_loss)

    return models, optimizers, schedulers, early_stoppings, losses


# 训练多任务学习的 FCNN 模型
def train_mtl_fcnn(models, optimizers, schedulers, early_stoppings, losses, train_loader, device, num_epochs=300):
    mtl_model, _, _, _, _, _, _ = models
    optimizer = optimizers[0]
    scheduler = schedulers[0]
    early_stopping = early_stoppings[0]
    mse_loss, focal_loss = losses

    loss_history_clf = []
    loss_history_reg = []
    scaler = GradScaler()

    for epoch in range(num_epochs):
        mtl_model.train()
        running_loss_clf = 0.0
        running_loss_reg = 0.0
        for X_batch, y_batch_clf, y_batch_reg in train_loader:
            X_batch = X_batch.to(device)
            y_batch_clf = y_batch_clf.to(device)
            y_batch_reg = y_batch_reg.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs_clf, outputs_reg = mtl_model(X_batch)
                loss_reg = mse_loss(outputs_reg, y_batch_reg)
                loss_clf = focal_loss(outputs_clf, y_batch_clf)
                # 组合损失（可以根据需要调整权重）
                loss = loss_reg + loss_clf
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss_reg += loss_reg.item()
            running_loss_clf += loss_clf.item()

        # 计算平均损失
        avg_loss_reg = running_loss_reg / len(train_loader)
        avg_loss_clf = running_loss_clf / len(train_loader)
        loss_history_reg.append(avg_loss_reg)
        loss_history_clf.append(avg_loss_clf)

        # 打印和记录损失
        print(f"Epoch {epoch + 1} - Regression Loss: {avg_loss_reg:.4f}, Classification Loss: {avg_loss_clf:.4f}")
        logging.info(
            f"Epoch {epoch + 1} - Regression Loss: {avg_loss_reg:.4f}, Classification Loss: {avg_loss_clf:.4f}")

        # 步进学习率调度器
        scheduler.step()

        # 检查早停
        early_stopping(avg_loss_reg + avg_loss_clf)  # 可以根据需要调整
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            logging.info("Early stopping triggered.")
            break

    return loss_history_reg, loss_history_clf


# 绘制损失曲线
def plot_loss_curve(loss_history_reg, loss_history_clf, fold, task='multi-task', eval=False):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history_reg, label='Regression Loss', color='blue', linestyle='-')
    plt.plot(loss_history_clf, label='Classification Loss', color='orange', linestyle='--')
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
    plt.savefig(filename)  # 保存绘图
    plt.close()  # 关闭绘图以避免显示问题


# 生成多任务学习的元特征（回归和分类）
def generate_meta_features_mtl(models, test_loader, X_test_scaled, device):
    mtl_model, lgbm_reg, lgbm_clf, xgb_reg, xgb_clf, rf_reg, rf_clf = models
    mtl_model.eval()
    preds_reg, preds_clf = [], []
    with torch.no_grad():
        for X_batch, _, _ in test_loader:
            X_batch = X_batch.to(device)
            output_clf, output_reg = mtl_model(X_batch)
            preds_reg.extend(output_reg.cpu().numpy())
            preds_clf.extend(torch.softmax(output_clf, dim=1).cpu().numpy())  # 分类概率

    # LightGBM 的预测
    lgbm_reg_preds = lgbm_reg.predict(X_test_scaled)
    lgbm_clf_probs = lgbm_clf.predict_proba(X_test_scaled)

    # XGBoost 的预测
    xgb_reg_preds = xgb_reg.predict(X_test_scaled)
    xgb_clf_probs = xgb_clf.predict_proba(X_test_scaled)

    # 随机森林的预测
    rf_reg_preds = rf_reg.predict(X_test_scaled)
    rf_clf_probs = rf_clf.predict_proba(X_test_scaled)

    # 组合所有预测作为元特征
    fold_meta_features_reg = np.column_stack([
        preds_reg,
        lgbm_reg_preds,
        xgb_reg_preds,
        rf_reg_preds
    ])

    fold_meta_features_clf = np.hstack([
        np.array(preds_clf),
        lgbm_clf_probs,
        xgb_clf_probs,
        rf_clf_probs
    ])

    return fold_meta_features_reg, fold_meta_features_clf


# 调整回归预测以确保与分类预测一致
def adjust_regression_with_classification(reg_output, clf_output):
    """
    用分类模型的预测结果调整回归模型的输出
    :param reg_output: 回归模型预测的价格变化量
    :param clf_output: 分类模型预测的涨跌概率
    :return: 调整后的回归输出
    """
    # 分类结果：0 表示跌，1 表示涨
    clf_prediction = (clf_output > 0.5).float()

    # 符号检查
    reg_sign = torch.sign(reg_output)
    clf_sign = torch.sign(clf_prediction - 0.5)  # 分类的涨跌方向

    # 如果符号不一致，调整回归结果的符号
    adjusted_reg_output = torch.where(
        reg_sign != clf_sign,  # 如果方向不一致
        -reg_output,  # 翻转回归预测
        reg_output  # 保持原预测
    )

    return adjusted_reg_output


# 强制回归预测值与分类预测结果一致
def enforce_consistency(reg_output, clf_output):
    """
    强制回归预测值与分类预测结果一致
    :param reg_output: 回归模型的预测值
    :param clf_output: 分类模型的预测概率
    :return: 一致性校正后的回归预测值
    """
    clf_prediction = (clf_output > 0.5).float()  # 分类预测结果

    # 如果分类预测涨，修正回归预测为正
    reg_output = torch.where(
        clf_prediction == 1,
        torch.relu(reg_output),  # 保证回归值为正
        reg_output
    )

    # 如果分类预测跌，修正回归预测为负
    reg_output = torch.where(
        clf_prediction == 0,
        -torch.relu(-reg_output),  # 保证回归值为负
        reg_output
    )

    return reg_output


# 主函数
def main():
    # 文件路径（根据需要修改）
    file_path = "/home/w/桌面/lilj/GOLD_cleaned.xlsx"
    try:
        X, y_class, y_reg = preprocess_data(file_path)
    except Exception as e:
        print(f"Error in preprocessing data: {e}")
        logging.error(f"Error in preprocessing data: {e}")
        return

    # 检查目标分布
    print("Classification Target Distribution:")
    print(y_class.value_counts())
    logging.info(f"Classification Target Distribution:\n{y_class.value_counts()}")

    print("\nRegression Target Statistics:")
    print(y_reg.describe())
    logging.info(f"Regression Target Statistics:\n{y_reg.describe()}")

    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    logging.info(f"Using device: {device}")

    # 初始化时间序列分割
    tscv = TimeSeriesSplit(n_splits=5)

    # 存储元特征和标签的列表
    meta_train_features_reg = []
    meta_train_labels_reg = []

    meta_train_features_clf = []
    meta_train_labels_clf = []

    # 存储评估指标
    # 回归指标
    rmses, mapes = [], []

    # 分类指标
    accuracies, precisions, recalls, f1_scores, aucs = [], [], [], [], []

    # 第一次交叉验证：生成回归和分类的元特征并训练基础模型
    print("\nGenerating meta-features and training base models for Regression and Classification...")
    logging.info("Generating meta-features and training base models started.")

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\nFold {fold + 1} - Training base models")
        logging.info(f"Fold {fold + 1} - Training base models started")

        # 分割数据
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train_clf_fold, y_test_clf_fold = y_class.iloc[train_index], y_class.iloc[test_index]
        y_train_reg_fold, y_test_reg_fold = y_reg.iloc[train_index], y_reg.iloc[test_index]

        # 特征缩放
        X_train_scaled, X_test_scaled = preprocess_and_scale(X_train, X_test, scaler_type='StandardScaler')

        # 创建多任务学习的数据集和数据加载器
        train_dataset = FinancialMultiTaskDataset(X_train_scaled, y_train_clf_fold, y_train_reg_fold)
        test_dataset = FinancialMultiTaskDataset(X_test_scaled, y_test_clf_fold, y_test_reg_fold)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 训练多任务学习的基础模型
        models_mtl, optimizers_mtl, schedulers_mtl, early_stoppings_mtl, losses_mtl = train_base_models_mtl(
            X_train_scaled, y_train_clf_fold, y_train_reg_fold, device
        )
        loss_history_reg, loss_history_clf = train_mtl_fcnn(
            models=models_mtl,
            optimizers=optimizers_mtl,
            schedulers=schedulers_mtl,
            early_stoppings=early_stoppings_mtl,
            losses=losses_mtl,
            train_loader=train_loader,
            device=device
        )

        # 绘制多任务学习的损失曲线
        plot_loss_curve(loss_history_reg, loss_history_clf, fold, task='multi-task')

        # 获取所有模型
        mtl_model, lgbm_reg, lgbm_clf, xgb_reg, xgb_clf, rf_reg, rf_clf = models_mtl

        # 训练 LightGBM 回归器
        try:
            lgbm_reg.fit(X_train_scaled, y_train_reg_fold)
        except Exception as e:
            print(f"Error during LGBMRegressor training: {e}")
            logging.error(f"Error during LGBMRegressor training: {e}")
            return

        # 训练 LightGBM 分类器
        try:
            lgbm_clf.fit(X_train_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during LGBMClassifier training: {e}")
            logging.error(f"Error during LGBMClassifier training: {e}")
            return

        # 训练 XGBoost 回归器
        try:
            xgb_reg.fit(X_train_scaled, y_train_reg_fold)
        except Exception as e:
            print(f"Error during XGBoost Regressor training: {e}")
            logging.error(f"Error during XGBoost Regressor training: {e}")
            return

        # 训练 XGBoost 分类器
        try:
            xgb_clf.fit(X_train_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during XGBoost Classifier training: {e}")
            logging.error(f"Error during XGBoost Classifier training: {e}")
            return

        # 训练随机森林回归器
        try:
            rf_reg.fit(X_train_scaled, y_train_reg_fold)
        except Exception as e:
            print(f"Error during RandomForestRegressor training: {e}")
            logging.error(f"Error during RandomForestRegressor training: {e}")
            return

        # 训练随机森林分类器
        try:
            rf_clf.fit(X_train_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during RandomForestClassifier training: {e}")
            logging.error(f"Error during RandomForestClassifier training: {e}")
            return

        # 生成多任务学习的元特征
        fold_meta_features_reg, fold_meta_features_clf = generate_meta_features_mtl(
            models=models_mtl,
            test_loader=test_loader,
            X_test_scaled=X_test_scaled,  # 传递 X_test_scaled
            device=device
        )

        # 追加到元特征和标签列表中
        meta_train_features_reg.append(fold_meta_features_reg)
        meta_train_labels_reg.extend(y_test_reg_fold.values)

        meta_train_features_clf.append(fold_meta_features_clf)
        meta_train_labels_clf.extend(y_test_clf_fold.values)

    # 合并所有元特征和标签
    meta_train_features_reg = np.vstack(meta_train_features_reg)
    meta_train_labels_reg = np.array(meta_train_labels_reg)

    meta_train_features_clf = np.vstack(meta_train_features_clf)
    meta_train_labels_clf = np.array(meta_train_labels_clf)

    # 训练回归的 Meta-Model
    print("\nTraining Meta-Model for Regression...")
    logging.info("Training Meta-Model for Regression started.")
    meta_model_reg = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.15,
        max_depth=5,
        random_state=42
    )
    meta_model_reg.fit(meta_train_features_reg, meta_train_labels_reg)

    # 训练分类的 Meta-Model
    print("\nTraining Meta-Model for Classification...")
    logging.info("Training Meta-Model for Classification started.")
    meta_model_clf = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.15,
        max_depth=5,
        random_state=42
    )
    meta_model_clf.fit(meta_train_features_clf, meta_train_labels_clf)

    # 第二次交叉验证：评估集成模型
    print("\nEvaluating Ensemble Models for Regression and Classification...")
    logging.info("Evaluating Ensemble Models started.")

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\nFold {fold + 1} - Evaluating Ensemble Models")
        logging.info(f"Fold {fold + 1} - Evaluating Ensemble Models started")

        # 分割数据
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train_clf_fold, y_test_clf_fold = y_class.iloc[train_index], y_class.iloc[test_index]
        y_train_reg_fold, y_test_reg_fold = y_reg.iloc[train_index], y_reg.iloc[test_index]

        # 特征缩放
        X_train_scaled, X_test_scaled = preprocess_and_scale(X_train, X_test, scaler_type='StandardScaler')

        # 创建多任务学习的数据集和数据加载器
        train_dataset = FinancialMultiTaskDataset(X_train_scaled, y_train_clf_fold, y_train_reg_fold)
        test_dataset = FinancialMultiTaskDataset(X_test_scaled, y_test_clf_fold, y_test_reg_fold)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 训练多任务学习的基础模型
        models_mtl, optimizers_mtl, schedulers_mtl, early_stoppings_mtl, losses_mtl = train_base_models_mtl(
            X_train_scaled, y_train_clf_fold, y_train_reg_fold, device
        )
        loss_history_reg, loss_history_clf = train_mtl_fcnn(
            models=models_mtl,
            optimizers=optimizers_mtl,
            schedulers=schedulers_mtl,
            early_stoppings=early_stoppings_mtl,
            losses=losses_mtl,
            train_loader=train_loader,
            device=device
        )

        # 绘制多任务学习的损失曲线
        plot_loss_curve(loss_history_reg, loss_history_clf, fold, task='multi-task', eval=True)

        # 获取所有模型
        mtl_model, lgbm_reg, lgbm_clf, xgb_reg, xgb_clf, rf_reg, rf_clf = models_mtl

        # 训练 LightGBM 回归器
        try:
            lgbm_reg.fit(X_train_scaled, y_train_reg_fold)
        except Exception as e:
            print(f"Error during LGBMRegressor training: {e}")
            logging.error(f"Error during LGBMRegressor training: {e}")
            return

        # 训练 LightGBM 分类器
        try:
            lgbm_clf.fit(X_train_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during LGBMClassifier training: {e}")
            logging.error(f"Error during LGBMClassifier training: {e}")
            return

        # 训练 XGBoost 回归器
        try:
            xgb_reg.fit(X_train_scaled, y_train_reg_fold)
        except Exception as e:
            print(f"Error during XGBoost Regressor training: {e}")
            logging.error(f"Error during XGBoost Regressor training: {e}")
            return

        # 训练 XGBoost 分类器
        try:
            xgb_clf.fit(X_train_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during XGBoost Classifier training: {e}")
            logging.error(f"Error during XGBoost Classifier training: {e}")
            return

        # 训练随机森林回归器
        try:
            rf_reg.fit(X_train_scaled, y_train_reg_fold)
        except Exception as e:
            print(f"Error during RandomForestRegressor training: {e}")
            logging.error(f"Error during RandomForestRegressor training: {e}")
            return

        # 训练随机森林分类器
        try:
            rf_clf.fit(X_train_scaled, y_train_clf_fold)
        except Exception as e:
            print(f"Error during RandomForestClassifier training: {e}")
            logging.error(f"Error during RandomForestClassifier training: {e}")
            return

        # 生成多任务学习的元特征
        fold_meta_features_reg, fold_meta_features_clf = generate_meta_features_mtl(
            models=models_mtl,
            test_loader=test_loader,
            X_test_scaled=X_test_scaled,  # 传递 X_test_scaled
            device=device
        )

        # 使用 Meta-Model 进行预测
        fold_meta_preds_reg = meta_model_reg.predict(fold_meta_features_reg)
        fold_meta_preds_clf = meta_model_clf.predict(fold_meta_features_clf)
        fold_meta_probs_clf = meta_model_clf.predict_proba(fold_meta_features_clf)[:, 1]

        # 计算回归指标
        rmse = root_mean_squared_error(y_test_reg_fold, fold_meta_preds_reg)
        mape = mean_absolute_percentage_error_custom(y_test_reg_fold, fold_meta_preds_reg)

        # 存储回归指标
        rmses.append(rmse)
        mapes.append(mape)

        # 打印和记录回归指标
        print(f"Fold {fold + 1} - Regression: RMSE (USD/oz)={rmse:.4f}, MAPE (%)={mape:.4f}")
        logging.info(f"Fold {fold + 1} - Regression: RMSE (USD/oz)={rmse:.4f}, MAPE (%)={mape:.4f}")

        # 计算分类指标
        acc = accuracy_score(y_test_clf_fold, fold_meta_preds_clf)
        prec = precision_score(y_test_clf_fold, fold_meta_preds_clf)
        rec = recall_score(y_test_clf_fold, fold_meta_preds_clf)
        f1 = f1_score(y_test_clf_fold, fold_meta_preds_clf)
        auc = roc_auc_score(y_test_clf_fold, fold_meta_probs_clf)

        # 存储分类指标
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        aucs.append(auc)

        # 打印和记录分类指标
        print(
            f"Fold {fold + 1} - Classification: Accuracy={acc:.4f}, Precision={prec:.4f}, "
            f"Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}"
        )
        logging.info(
            f"Fold {fold + 1} - Classification: Accuracy={acc:.4f}, Precision={prec:.4f}, "
            f"Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}"
        )

    # 汇总回归的整体结果
    print("\nOverall Regression Results:")
    print(f"Average RMSE (USD/oz): {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"Average MAPE (%): {np.mean(mapes):.4f} ± {np.std(mapes):.4f}")
    logging.info(
        f"Overall Regression Results: "
        f"Average RMSE (USD/oz)={np.mean(rmses):.4f} ± {np.std(rmses):.4f}, "
        f"Average MAPE (%)={np.mean(mapes):.4f} ± {np.std(mapes):.4f}"
    )

    # 汇总分类的整体结果
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

    # 保存整体结果
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
