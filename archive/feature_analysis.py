import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, mean_squared_error, mean_absolute_percentage_error
)
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ta
import warnings
from torch.utils.data import Dataset, DataLoader
import logging
import joblib
import sys

# 关闭警告
warnings.filterwarnings('ignore')  # Suppress warnings

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 配置日志记录
logging.basicConfig(
    filename='feature_analysis_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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

# 定义回归元特征生成函数
def generate_meta_features_reg(fcnn2, lgbm_reg, xgb_reg, rf_reg, test_loader, X_train_scaled, y_train_reg, X_test_scaled, device):
    try:
        # 确保 fcnn2 是已训练的模型
        fcnn2.eval()
        preds2 = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                output2 = fcnn2(X_batch).squeeze().cpu().numpy()
                preds2.extend(output2)

        # 检查 preds2 是否为空
        if len(preds2) == 0:
            raise ValueError("FCNNRegressor 没有生成任何预测。")

        # 训练 LightGBM 回归器在训练数据上
        lgbm_reg.fit(X_train_scaled, y_train_reg)

        # 训练 XGBoost 回归器在训练数据上
        xgb_reg.fit(X_train_scaled, y_train_reg)

        # 训练随机森林回归器在训练数据上
        rf_reg.fit(X_train_scaled, y_train_reg)

        # 使用训练好的模型生成预测
        lgbm_preds = lgbm_reg.predict(X_test_scaled)
        xgb_preds = xgb_reg.predict(X_test_scaled)
        rf_preds = rf_reg.predict(X_test_scaled)

        # 检查预测结果长度是否匹配
        if not (len(lgbm_preds) == len(xgb_preds) == len(rf_preds) == len(preds2)):
            raise ValueError("不同模型的预测长度不匹配。")

        # 堆叠所有模型预测生成元特征
        fold_meta_features = np.column_stack([preds2, lgbm_preds, xgb_preds, rf_preds])
        return fold_meta_features
    except Exception as e:
        logging.error(f"Error in generate_meta_features_reg: {e}")
        sys.exit(1)

# 定义分类元特征生成函数
def generate_meta_features_clf(fcnn2, lgbm_clf, xgb_clf, rf_clf, test_loader, X_train_scaled, y_train_clf, X_test_scaled, device):
    try:
        # 确保 fcnn2 是已训练的模型
        fcnn2.eval()
        preds2 = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                output2 = fcnn2(X_batch).cpu().numpy()
                preds2.extend(output2)

        # 将 preds2 转换为概率
        preds2 = 1 / (1 + np.exp(-np.array(preds2)))  # Sigmoid

        # 将概率转换为标签
        preds2_labels = (preds2 > 0.5).astype(int)

        # 检查 preds2_labels 是否为空
        if len(preds2_labels) == 0:
            raise ValueError("FCNNClassifier 没有生成任何预测标签。")

        # 训练 LightGBM 分类器在训练数据上
        lgbm_clf.fit(X_train_scaled, y_train_clf)

        # 训练 XGBoost 分类器在训练数据上
        xgb_clf.fit(X_train_scaled, y_train_clf)

        # 训练随机森林分类器在训练数据上
        rf_clf.fit(X_train_scaled, y_train_clf)

        # 使用训练好的模型生成概率预测
        lgbm_preds = lgbm_clf.predict_proba(X_test_scaled)[:, 1]
        xgb_preds = xgb_clf.predict_proba(X_test_scaled)[:, 1]
        rf_preds = rf_clf.predict_proba(X_test_scaled)[:, 1]

        # 检查预测结果长度是否匹配
        if not (len(lgbm_preds) == len(xgb_preds) == len(rf_preds) == len(preds2)):
            raise ValueError("不同模型的预测长度不匹配。")

        # 堆叠所有模型预测生成元特征
        fold_meta_features = np.column_stack([preds2, lgbm_preds, xgb_preds, rf_preds])
        return fold_meta_features
    except Exception as e:
        logging.error(f"Error in generate_meta_features_clf: {e}")
        sys.exit(1)

# 定义回归数据集类
class FinancialRegressionDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets.values.astype(np.float32)  # 确保 targets 是 float 类型用于回归

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )

# 定义分类数据集类
class FinancialClassificationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels.values.astype(np.int64)  # 确保 labels 是 int64 类型用于分类

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# 定义回归用的全连接神经网络模型
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
        return self.network(x)  # 原始的连续输出

# 定义分类用的全连接神经网络模型
class FCNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout=0.5):
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
        return self.network(x)  # 原始的 logits 输出

# 定义均方根误差 (RMSE)
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# 定义平均绝对百分比误差 (MAPE)
def mean_absolute_percentage_error_custom(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100.0

# 定义数据预处理函数
def preprocess_data(file_path):
    try:
        # 加载数据
        data = pd.read_excel(file_path)

        # 确保 'Date' 列是 datetime 类型
        if not np.issubdtype(data['Date'].dtype, np.datetime64):
            data['Date'] = pd.to_datetime(data['Date'])

        # 创建分类目标：如果第二天的 GOLD 价格上涨，则为1，否则为0
        data['Next_Day_Change'] = (data['GOLD'].diff() > 0).astype(int)

        # 创建回归目标：第二天的 GOLD 价格
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
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 7 else np.nan, raw=True
        )

        # 通胀调整后的黄金价格
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

        # 将今天的特征提前一天，以避免未来数据泄漏
        today_features = [
            'Gold_Oil_Ratio', 'USD_CNY_to_JPY', 'China_US_CPI_Ratio',
            'Gold_Rate_of_Change', 'Oil_Rate_of_Change', 'Gold_Trend_7_days',
            'Gold_Inflation_Adjusted', 'Gold_Volatility_10_days', 'Oil_Volatility_10_days',
            'Gold_SP500_Ratio', 'US_Japan_Interest_Rate_Diff'
        ]
        data[today_features] = data[today_features].shift(1)

        # 数据清理：移除包含 NaN 或 Inf 的行
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        # 确保所有特征都存在
        missing_features = [feature for feature in features if feature not in data.columns]
        if missing_features:
            raise ValueError(f"The following required features are missing from the data: {missing_features}")

        # 分离特征和目标变量
        X = data[features]
        y_class = data['Next_Day_Change']
        y_reg = data['Next_Day_Price']

        logging.info("Data preprocessing completed successfully.")
        return X, y_class, y_reg, features

    except Exception as e:
      print(f"An error occurred: {e}")

# 回归特征消融分析
def feature_ablation_regression(meta_model_reg, X_test_meta, y_test_reg, meta_feature_names_reg):
    try:
        # 计算基线 RMSE
        baseline_preds = meta_model_reg.predict(X_test_meta)
        baseline_rmse = root_mean_squared_error(y_test_reg, baseline_preds)

        feature_importances = []

        num_features = X_test_meta.shape[1]
        min_features = 1  # 设置最小特征数为1，避免特征过少

        # 遍历每个特征，进行消融测试
        for i in range(num_features):
            if num_features - 1 < min_features:
                logging.warning(f"Feature ablation stopped. Minimum feature requirement of {min_features} reached.")
                break

            # 移除当前特征
            X_modified = np.delete(X_test_meta, i, axis=1)

            # 重新训练一个新的回归模型，以匹配新维度
            ablation_model = GradientBoostingRegressor(
                n_estimators=meta_model_reg.n_estimators,
                learning_rate=meta_model_reg.learning_rate,
                max_depth=meta_model_reg.max_depth,
                random_state=meta_model_reg.random_state
            )
            ablation_model.fit(X_modified, y_test_reg)

            # 计算消融后的 RMSE
            preds_modified = ablation_model.predict(X_modified)
            rmse_modified = root_mean_squared_error(y_test_reg, preds_modified)

            # 特征重要性为消融后的 RMSE 与基线 RMSE 的差值
            importance = rmse_modified - baseline_rmse
            feature_importances.append(importance)

        # 获取当前特征列表
        current_features = meta_feature_names_reg[:len(feature_importances)]

        # 将结果转换为 DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': current_features,
            'Importance': feature_importances
        })
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        logging.info("Feature ablation for regression completed successfully.")
        return feature_importance_df
    except Exception as e:
        logging.error(f"Error in feature_ablation_regression: {e}")
        sys.exit(1)

# 分类特征消融分析
def feature_ablation_classification(meta_model_clf, X_test_meta, y_test_clf, meta_feature_names_clf):
    try:
        # 计算基线 AUC
        baseline_auc = roc_auc_score(y_test_clf, meta_model_clf.predict_proba(X_test_meta)[:, 1])

        feature_importances = []

        num_features = X_test_meta.shape[1]
        min_features = 1  # 设置最小特征数为1，避免特征过少

        # 遍历每个特征，进行消融测试
        for i in range(num_features):
            if num_features - 1 < min_features:
                logging.warning(f"Feature ablation stopped. Minimum feature requirement of {min_features} reached.")
                break

            # 移除当前特征
            X_modified = np.delete(X_test_meta, i, axis=1)

            # 重新训练一个新的分类模型，以匹配新维度
            ablation_model = GradientBoostingClassifier(
                n_estimators=meta_model_clf.n_estimators,
                learning_rate=meta_model_clf.learning_rate,
                max_depth=meta_model_clf.max_depth,
                random_state=meta_model_clf.random_state
            )
            ablation_model.fit(X_modified, y_test_clf)

            # 计算消融后的 AUC
            preds_modified = ablation_model.predict_proba(X_modified)[:, 1]
            auc_modified = roc_auc_score(y_test_clf, preds_modified)

            # 特征重要性为基线 AUC 与消融后的 AUC 差值
            importance = baseline_auc - auc_modified
            feature_importances.append(importance)

        # 获取当前特征列表
        current_features = meta_feature_names_clf[:len(feature_importances)]

        # 将结果转换为 DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': current_features,
            'Importance': feature_importances
        })
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        logging.info("Feature ablation for classification completed successfully.")
        return feature_importance_df
    except Exception as e:
        logging.error(f"Error in feature_ablation_classification: {e}")
        sys.exit(1)

# 绘制特征重要性
def plot_feature_importance(feature_importance_df, task='regression'):
    try:
        # 按重要性排序
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

        plt.figure(figsize=(10, max(6, len(feature_importance_df)*0.3)))  # 动态调整高度

        # 定义颜色
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

# 主执行函数
def main():
    try:
        # 数据预处理
        file_path = "/home/w/桌面/lilj/GOLD_cleaned.xlsx"  # 请根据实际路径修改
        X, y_class, y_reg, feature_list = preprocess_data(file_path)

        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, 'scaler.joblib')  # 保存缩放器以备后用

        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        fold = 1

        # 初始化列表以存储各折的评估结果
        regression_results = []
        classification_results = []

        # 初始化元特征
        meta_features_reg = []
        meta_features_clf = []
        meta_labels_reg = []
        meta_labels_clf = []

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")

        for train_index, test_index in tscv.split(X_scaled):
            logging.info(f"Processing fold {fold}...")
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train_reg, y_test_reg_fold = y_reg.iloc[train_index], y_reg.iloc[test_index]
            y_train_clf, y_test_clf_fold = y_class.iloc[train_index], y_class.iloc[test_index]

            # 创建数据集和数据加载器
            reg_dataset = FinancialRegressionDataset(X_train, y_train_reg)
            reg_loader = DataLoader(reg_dataset, batch_size=64, shuffle=True)

            clf_dataset = FinancialClassificationDataset(X_train, y_train_clf)
            clf_loader = DataLoader(clf_dataset, batch_size=64, shuffle=True)

            # 初始化模型
            input_dim = X_train.shape[1]
            hidden_dims = [128, 64, 32]

            fcnn_reg = FCNNRegressor(input_dim, hidden_dims).to(device)
            fcnn_clf = FCNNClassifier(input_dim, hidden_dims, output_dim=1).to(device)  # 修改输出维度为1，用于二分类

            # 定义损失函数和优化器
            criterion_reg = nn.MSELoss()
            optimizer_reg = torch.optim.Adam(fcnn_reg.parameters(), lr=0.001)

            criterion_clf = nn.BCEWithLogitsLoss()  # 使用 BCEWithLogitsLoss 适用于二分类
            optimizer_clf = torch.optim.Adam(fcnn_clf.parameters(), lr=0.001)

            # 训练回归模型
            fcnn_reg.train()
            for epoch in range(50):  # 你可以调整 epoch 数量
                epoch_losses = []
                for batch_X, batch_y in reg_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer_reg.zero_grad()
                    outputs = fcnn_reg(batch_X).squeeze()
                    loss = criterion_reg(outputs, batch_y)
                    loss.backward()
                    optimizer_reg.step()
                    epoch_losses.append(loss.item())
                avg_loss = np.mean(epoch_losses)
                if (epoch + 1) % 10 == 0:
                    logging.info(f"Fold {fold}, Regression Epoch {epoch+1}, Loss: {avg_loss:.4f}")

            # 训练分类模型
            fcnn_clf.train()
            for epoch in range(50):  # 你可以调整 epoch 数量
                epoch_losses = []
                for batch_X, batch_y in clf_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer_clf.zero_grad()
                    outputs = fcnn_clf(batch_X).squeeze()
                    loss = criterion_clf(outputs, batch_y.float())  # 转换为 float 类型
                    loss.backward()
                    optimizer_clf.step()
                    epoch_losses.append(loss.item())
                avg_loss = np.mean(epoch_losses)
                if (epoch + 1) % 10 == 0:
                    logging.info(f"Fold {fold}, Classification Epoch {epoch+1}, Loss: {avg_loss:.4f}")

            # 生成元特征
            test_loader_reg = DataLoader(FinancialRegressionDataset(X_test, y_test_reg_fold), batch_size=64, shuffle=False)
            test_loader_clf = DataLoader(FinancialClassificationDataset(X_test, y_test_clf_fold), batch_size=64, shuffle=False)

            # 生成回归元特征
            meta_feat_reg = generate_meta_features_reg(
                fcnn_reg,
                GradientBoostingRegressor(),
                GradientBoostingRegressor(),
                RandomForestRegressor(),
                test_loader_reg,
                X_train,
                y_train_reg,
                X_test,
                device
            )
            meta_features_reg.append(meta_feat_reg)
            meta_labels_reg.extend(y_test_reg_fold)

            # 生成分类元特征
            meta_feat_clf = generate_meta_features_clf(
                fcnn_clf,
                GradientBoostingClassifier(),
                GradientBoostingClassifier(),
                RandomForestClassifier(),
                test_loader_clf,
                X_train,
                y_train_clf,
                X_test,
                device
            )
            meta_features_clf.append(meta_feat_clf)
            meta_labels_clf.extend(y_test_clf_fold)

            fold += 1

        # 合并所有折的元特征
        X_meta_reg = np.vstack(meta_features_reg)
        y_meta_reg = np.array(meta_labels_reg)

        X_meta_clf = np.vstack(meta_features_clf)
        y_meta_clf = np.array(meta_labels_clf)

        # 定义元特征名称
        meta_feature_names_reg = ['FCNN_Pred', 'LightGBM_Pred', 'XGBoost_Pred', 'RandomForest_Pred']
        meta_feature_names_clf = ['FCNN_Prob', 'LightGBM_Prob', 'XGBoost_Prob', 'RandomForest_Prob']

        # 检查元特征的维度是否满足最小要求
        # 设置最小特征数为1，确保有足够的特征
        if X_meta_reg.shape[1] < 1 or X_meta_clf.shape[1] < 1:
            raise ValueError("Meta features 数量少于1，GBM 可能无法正常训练。请检查特征生成和消融过程。")

        # 标准化元特征
        scaler_meta_reg = StandardScaler()
        X_meta_reg_scaled = scaler_meta_reg.fit_transform(X_meta_reg)
        joblib.dump(scaler_meta_reg, 'scaler_meta_reg.joblib')

        scaler_meta_clf = StandardScaler()
        X_meta_clf_scaled = scaler_meta_clf.fit_transform(X_meta_clf)
        joblib.dump(scaler_meta_clf, 'scaler_meta_clf.joblib')

        # 训练最终的元模型
        # 回归
        meta_model_reg = GradientBoostingRegressor()
        meta_model_reg.fit(X_meta_reg_scaled, y_meta_reg)
        joblib.dump(meta_model_reg, 'meta_model_reg.joblib')

        # 分类
        meta_model_clf = GradientBoostingClassifier()
        meta_model_clf.fit(X_meta_clf_scaled, y_meta_clf)
        joblib.dump(meta_model_clf, 'meta_model_clf.joblib')

        # 回归评估
        y_pred_reg = meta_model_reg.predict(X_meta_reg_scaled)
        rmse = root_mean_squared_error(y_meta_reg, y_pred_reg)
        mape = mean_absolute_percentage_error_custom(y_meta_reg, y_pred_reg)
        logging.info(f"Meta Model Regression RMSE (USD/oz): {rmse:.4f}")
        logging.info(f"Meta Model Regression MAPE (%): {mape:.4f}")
        regression_results.append({'RMSE': rmse, 'MAPE': mape})

        # 分类评估
        y_pred_clf_proba = meta_model_clf.predict_proba(X_meta_clf_scaled)[:,1]
        auc = roc_auc_score(y_meta_clf, y_pred_clf_proba)
        logging.info(f"Meta Model Classification AUC: {auc:.4f}")
        classification_results.append({'AUC': auc})

        # 特征重要性分析
        feature_importance_reg = feature_ablation_regression(meta_model_reg, X_meta_reg_scaled, y_meta_reg, meta_feature_names_reg)
        plot_feature_importance(feature_importance_reg, task='regression')

        feature_importance_clf = feature_ablation_classification(meta_model_clf, X_meta_clf_scaled, y_meta_clf, meta_feature_names_clf)
        plot_feature_importance(feature_importance_clf, task='classification')

        # 记录最终结果
        logging.info(f"Final Regression Results: {regression_results}")
        logging.info(f"Final Classification Results: {classification_results}")

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
