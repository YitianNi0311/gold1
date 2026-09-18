import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_percentage_error
)
import warnings
import ta  # 技术指标库

warnings.filterwarnings('ignore')  # Suppress warnings

# 数据预处理函数
def preprocess_data(file_path):
    data = pd.read_excel(file_path)

    if not np.issubdtype(data['Date'].dtype, np.datetime64):
        data['Date'] = pd.to_datetime(data['Date'])

    # 创建分类目标和回归目标
    data['Next_Day_Change'] = (data['GOLD'].diff() > 0).astype(int)
    data['Next_Day_Gold_Price'] = data['GOLD'].shift(-1)

    # 特征工程
    for lag in range(1, 31):
        data[f'GOLD_lag{lag}'] = data['GOLD'].shift(lag)

    data['GOLD_MA_3_days'] = data['GOLD'].rolling(window=3).mean()
    data['GOLD_MA_5_days'] = data['GOLD'].rolling(window=5).mean()
    data['GOLD_MA_10_days'] = data['GOLD'].rolling(window=10).mean()

    data['RSI'] = ta.momentum.RSIIndicator(close=data['GOLD'], window=14).rsi()
    macd = ta.trend.MACD(close=data['GOLD'])
    data['MACD'] = macd.macd_diff()

    bollinger = ta.volatility.BollingerBands(close=data['GOLD'], window=20)
    data['Bollinger_High'] = bollinger.bollinger_hband()
    data['Bollinger_Low'] = bollinger.bollinger_lband()

    # 滚动特征
    data['GOLD_roll_mean_5'] = data['GOLD'].rolling(window=10).mean()
    data['GOLD_roll_std_5'] = data['GOLD'].rolling(window=10).std()
    data['GOLD_roll_min_5'] = data['GOLD'].rolling(window=10).min()
    data['GOLD_roll_max_5'] = data['GOLD'].rolling(window=10).max()

    # 自定义特征
    data['Gold_Oil_Ratio'] = data['GOLD'] / data['CrudeOil_SpotPrice_BrentUK']
    data['USD_CNY_to_JPY'] = data['SpotRate_USD_CNY'] / data['SpotRate_Tokyo_9AM_USD_JPY']
    data['China_US_CPI_Ratio'] = data['China_CPI_YoY_CurrentMonth'] / data['US_CPI_YoY_NSA']
    data['Gold_Rate_of_Change'] = data['GOLD'].pct_change(periods=5)
    data['Oil_Rate_of_Change'] = data['CrudeOil_SpotPrice_BrentUK'].pct_change(periods=5)
    data['Gold_Trend_7_days'] = data['GOLD'].rolling(window=7).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )
    data['Gold_Inflation_Adjusted'] = data['GOLD'] / data['US_CPI_YoY_NSA']
    data['Gold_Volatility_10_days'] = data['GOLD'].rolling(window=10).std()
    data['Oil_Volatility_10_days'] = data['CrudeOil_SpotPrice_BrentUK'].rolling(window=10).std()
    data['Gold_SP500_Ratio'] = data['GOLD'] / data['US_SP500_Index']
    data['US_Japan_Interest_Rate_Diff'] = data['US_DowJones_IndustrialAverage'] - data['SpotRate_Tokyo_9AM_USD_JPY']

    # 时间相关特征
    data['Month'] = data['Date'].dt.month
    data['Quarter'] = data['Date'].dt.quarter
    data['Day_of_Week'] = data['Date'].dt.dayofweek

    # 向前移动特定特征1天
    today_features = [
        'Gold_Oil_Ratio', 'USD_CNY_to_JPY', 'China_US_CPI_Ratio',
        'Gold_Rate_of_Change', 'Oil_Rate_of_Change', 'Gold_Trend_7_days',
        'Gold_Inflation_Adjusted', 'Gold_Volatility_10_days', 'Oil_Volatility_10_days',
        'Gold_SP500_Ratio', 'US_Japan_Interest_Rate_Diff'
    ]
    data[today_features] = data[today_features].shift(1)

    # 清理数据，删除 NaN 和 Inf 值
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    # 使用原始特征集合
    features = [
        'GOLD_MA_3_days', 'GOLD_lag1', 'GOLD_MA_5_days',
        'GOLD_lag2', 'GOLD_MA_10_days', 'GOLD_lag3', 'GOLD_lag4', 'GOLD_lag5', 'GOLD_lag6', 'GOLD_lag7', 'GOLD_lag8',
        'GOLD_lag9', 'GOLD_lag10',
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

    X = data[features]
    y_cls = data['Next_Day_Change']
    y_reg = data['Next_Day_Gold_Price']

    return X, y_cls, y_reg

class MultiTaskNN(nn.Module):
    def __init__(self, input_dim):
        super(MultiTaskNN, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.cls_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.reg_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        shared_rep = self.shared_layer(x)
        cls_out = self.cls_head(shared_rep)
        reg_out = self.reg_head(shared_rep)
        return cls_out, reg_out

def train_model(X_train, y_cls_train, y_reg_train, X_test, y_cls_test, y_reg_test, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_cls_train = torch.tensor(y_cls_train.values, dtype=torch.float32).unsqueeze(1).to(device)
    y_cls_test = torch.tensor(y_cls_test.values, dtype=torch.float32).unsqueeze(1).to(device)
    y_reg_train = torch.tensor(y_reg_train.values, dtype=torch.float32).unsqueeze(1).to(device)
    y_reg_test = torch.tensor(y_reg_test.values, dtype=torch.float32).unsqueeze(1).to(device)

    model = MultiTaskNN(X_train.size(1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion_cls = nn.BCELoss()
    criterion_reg = nn.MSELoss()

    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        cls_pred, reg_pred = model(X_train)
        loss_cls = criterion_cls(cls_pred, y_cls_train)
        loss_reg = criterion_reg(reg_pred, y_reg_train)
        loss = loss_cls + loss_reg
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        cls_pred, reg_pred = model(X_test)
        cls_pred_binary = (cls_pred.cpu().numpy().squeeze() > threshold).astype(int)
        acc = accuracy_score(y_cls_test.cpu().numpy(), cls_pred_binary)
        recall = recall_score(y_cls_test.cpu().numpy(), cls_pred_binary)
        precision = precision_score(y_cls_test.cpu().numpy(), cls_pred_binary)
        f1 = f1_score(y_cls_test.cpu().numpy(), cls_pred_binary)
        auc = roc_auc_score(y_cls_test.cpu().numpy(), cls_pred.cpu().numpy())
        rmse = np.sqrt(mean_squared_error(y_reg_test.cpu().numpy(), reg_pred.cpu().numpy()))
        mape = mean_absolute_percentage_error(y_reg_test.cpu().numpy(), reg_pred.cpu().numpy())

    return acc, recall, precision, f1, auc, rmse, mape

def main():
    file_path = '/home/w/桌面/lilj/GOLD_cleaned.xlsx'
    X, y_cls, y_reg = preprocess_data(file_path)

    tscv = TimeSeriesSplit(n_splits=5)

    results = {
        "accuracy": [], "recall": [], "precision": [], "f1": [], "auc": [], "rmse": [], "mape": []
    }

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"Fold {fold + 1}")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_cls_train, y_cls_test = y_cls.iloc[train_index], y_cls.iloc[test_index]
        y_reg_train, y_reg_test = y_reg.iloc[train_index], y_reg.iloc[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        acc, recall, precision, f1, auc, rmse, mape = train_model(
            X_train_scaled, y_cls_train, y_reg_train, X_test_scaled, y_cls_test, y_reg_test
        )

        results["accuracy"].append(acc)
        results["recall"].append(recall)
        results["precision"].append(precision)
        results["f1"].append(f1)
        results["auc"].append(auc)
        results["rmse"].append(rmse)
        results["mape"].append(mape)

        print(f"Fold {fold + 1} - Accuracy: {acc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        print(f"Fold {fold + 1} - RMSE: {rmse:.4f}, MAPE: {mape:.4f}")

    print("\nOverall Results:")
    for metric in results:
        print(f"{metric.capitalize()}: {np.mean(results[metric]):.4f} ± {np.std(results[metric]):.4f}")

if __name__ == "__main__":
    main()
