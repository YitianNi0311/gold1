import ta
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


# 数据预处理函数
def preprocess_data(file_path):
    # 加载数据
    data = pd.read_excel(file_path)

    # 确保 'Date' 列为 datetime 类型
    if not np.issubdtype(data['Date'].dtype, np.datetime64):
        data['Date'] = pd.to_datetime(data['Date'])

    # 创建下一个交易日的黄金价格作为目标变量（回归任务）
    data['Next_Day_Gold_Price'] = data['GOLD']

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

    rolling_window = 10
    data['GOLD_roll_mean_5'] = data['GOLD'].rolling(window=rolling_window).mean()
    data['GOLD_roll_std_5'] = data['GOLD'].rolling(window=rolling_window).std()
    data['GOLD_roll_min_5'] = data['GOLD'].rolling(window=rolling_window).min()
    data['GOLD_roll_max_5'] = data['GOLD'].rolling(window=rolling_window).max()

    # 自定义特征（向前移动1天）
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

    # 定义特征和目标（预测下一个交易日的黄金价格）
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

    # 分割特征和目标
    X = data[features]
    y = data['Next_Day_Gold_Price']

    return X, y


# PyTorch 神经网络模型
# PyTorch 神经网络模型
class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super(ANNModel, self).__init__()

        # 定义隐藏层
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))  # 第一层输入维度应为 input_dim
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout(dropout))

        # Subsequent hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))  # 修正为上一层的输出维度
            layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        # Combine layers into a module
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# 加载数据
file_path = '/home/w/桌面/lilj/GOLD_cleaned.xlsx'  # 替换为实际路径
X, y = preprocess_data(file_path)

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 转换为 PyTorch 张量
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32)

# 时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)

# 指标列表
rmse_list = []
mase_list = []

# 交叉验证循环
for train_idx, test_idx in tscv.split(X_tensor):
    X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
    y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]

    # 创建模型
    model = ANNModel(input_dim=X_train.shape[1], hidden_dims=[128, 64, 64], output_dim=1)

    # 损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练过程
    epochs = 300
    batch_size = 32
    early_stopping_patience = 20
    best_val_loss = np.inf
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Mini-batch 训练
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # 前向传播
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)  # 去掉多余的维度

            # 反向传播
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_test)
            val_loss = criterion(y_val_pred.squeeze(), y_test)

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # 预测
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).squeeze().numpy()  # 转换为 NumPy 数组

    # 计算 RMSE
    rmse = np.sqrt(mean_squared_error(y_test.numpy(), y_pred))  # y_test 转换为 NumPy 数组
    rmse_list.append(rmse)

    # 计算 MASE
    y_train_last = y_train[-1].item()  # 获取最后一个训练值
    y_baseline_pred = np.full_like(y_test.numpy(), y_train_last, dtype=np.float32)  # 确保是 NumPy 数组
    epsilon = 1e-8
    mase = np.mean(np.abs(y_test.numpy() - y_pred) / (np.abs(y_test.numpy() - y_baseline_pred) + epsilon))
    mase_list.append(mase)

# 输出结果
print(f"Average RMSE (USD/oz): {np.mean(rmse_list):.4f}")
print(f"Average MASE: {np.mean(mase_list):.4f}")
