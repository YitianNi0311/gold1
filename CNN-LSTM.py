import numpy as np
import pandas as pd
import ta
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import logging

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

class FinancialDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.y.iloc[index], dtype=torch.long)

# 假设 FinancialDataset 和 preprocess_data 等函数已在其他地方定义
# 你可以根据自己的需求调整这些函数

class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CNNLSTM, self).__init__()

        # 卷积层
        # 如果每个样本只有48个特征，input_size=1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1)  # input_size=1
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        # LSTM层
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=3, batch_first=True, dropout=0.3)

        # Dropout层
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x的shape是 (batch_size, seq_len)
        print(f"Input shape: {x.shape}")  # Debugging: Print input shape

        # 如果每个样本只有一个特征，则需要增加一个维度
        x = x.unsqueeze(1)  # 形状变为 (batch_size, 1, seq_len)
        print(f"Shape after unsqueeze: {x.shape}")  # Debugging: Print shape after unsqueeze

        # 卷积层
        x = self.conv1(x)  # input: (batch_size, 1, seq_len)
        x = nn.ReLU()(x)
        print(f"Shape after conv1: {x.shape}")  # Debugging: Print shape after conv1

        x = self.conv2(x)
        x = nn.ReLU()(x)
        print(f"Shape after conv2: {x.shape}")  # Debugging: Print shape after conv2

        x = self.conv3(x)
        x = nn.ReLU()(x)
        print(f"Shape after conv3: {x.shape}")  # Debugging: Print shape after conv3

        # Conv1d层的输出是 (batch_size, out_channels, seq_len)，转置为 (batch_size, seq_len, out_channels)
        x = x.transpose(1, 2)  # 转换为 (batch_size, seq_len, 256)
        print(f"Shape before LSTM: {x.shape}")  # Debugging: Print shape before LSTM

        # LSTM层
        x, (h_n, c_n) = self.lstm(x)  # 现在输入形状是 (batch_size, seq_len, 256)
        print(f"Shape after LSTM: {x.shape}")  # Debugging: Print shape after LSTM

        # 取LSTM最后一个时刻的输出
        x = x[:, -1, :]  # (batch_size, hidden_size)
        print(f"Shape after extracting last LSTM output: {x.shape}")  # Debugging: Print shape after extracting LSTM output

        # Dropout层
        x = self.dropout(x)

        # 返回LSTM最后一个时刻的输出
        return x



def main():
    # 文件路径
    file_path = "/home/w/桌面/lilj/GOLD_cleaned.xlsx"
    try:
        X, y = preprocess_data(file_path)
    except Exception as e:
        print(f"Error in preprocessing data: {e}")
        logging.error(f"Error in preprocessing data: {e}")
        return

    # 输出类别分布
    print("Class Distribution:")
    print(y.value_counts())
    logging.info(f"Class Distribution:\n{y.value_counts()}")

    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logging.info(f"Using device: {device}")

    # 初始化 TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    # 存储评估指标
    accuracies, precisions, recalls, f1_scores, aucs = [], [], [], [], []

    # 第一个交叉验证：训练 CNN-LSTM 模型
    print("Training CNN-LSTM model...")
    logging.info("Training CNN-LSTM model started.")

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"Fold {fold + 1} - Training CNN-LSTM")
        logging.info(f"Fold {fold + 1} - Training CNN-LSTM started")

        # 拆分数据
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 创建 Datasets 和 DataLoaders
        train_dataset = FinancialDataset(X_train_scaled, y_train)
        test_dataset = FinancialDataset(X_test_scaled, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 初始化模型
        model = CNNLSTM(input_size=X_train_scaled.shape[1], hidden_size=128, num_classes=2).to(device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

        # 训练模型
        try:
            model.train()
            for epoch in range(100):  # 训练10轮
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)

                    # 前向传播
                    optimizer.zero_grad()
                    output = model(data)

                    # 计算损失并反向传播
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

            print(f"Fold {fold + 1}: Training completed.")
        except Exception as e:
            print(f"Error during CNN-LSTM training: {e}")
            logging.error(f"Error during CNN-LSTM training: {e}")
            return

        # 评估模型
        model.eval()
        with torch.no_grad():
            y_pred = []
            y_prob = []
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = torch.argmax(output, dim=1)
                y_pred.extend(pred.cpu().numpy())
                y_prob.extend(torch.softmax(output, dim=1)[:, 1].cpu().numpy())

        # 计算评估指标
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # 存储评估指标
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        aucs.append(auc)

        # 输出和记录评估指标
        print(
            f"Fold {fold + 1}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
        logging.info(
            f"Fold {fold + 1}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    # 输出整体评估结果
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
