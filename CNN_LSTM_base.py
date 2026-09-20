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
    # load data
    data = pd.read_excel(file_path)

    # Date column -> datetime
    if not np.issubdtype(data['Date'].dtype, np.datetime64):
        data['Date'] = pd.to_datetime(data['Date'])

    # targets
    data['Next_Day_Change'] = (data['GOLD'].shift(-1) > data['GOLD']).astype(int)

    # feature engineering
    # lag features (history only, no shift needed)
    for lag in range(1, 31):
        data[f'GOLD_lag{lag}'] = data['GOLD'].shift(lag)

    # moving averages (history only, no shift needed)
    data['GOLD_MA_3_days'] = data['GOLD'].rolling(window=3).mean()
    data['GOLD_MA_5_days'] = data['GOLD'].rolling(window=5).mean()
    data['GOLD_MA_10_days'] = data['GOLD'].rolling(window=10).mean()

    # technical indicators (history only, no shift needed)
    data['RSI'] = ta.momentum.RSIIndicator(close=data['GOLD'], window=14).rsi()
    macd = ta.trend.MACD(close=data['GOLD'])
    data['MACD'] = macd.macd_diff()

    # bollinger bands (history only, no shift needed)
    bollinger = ta.volatility.BollingerBands(close=data['GOLD'], window=20)
    data['Bollinger_High'] = bollinger.bollinger_hband()
    data['Bollinger_Low'] = bollinger.bollinger_lband()

    # rolling stats (history only, no shift needed)
    rolling_window = 10
    data['GOLD_roll_mean_5'] = data['GOLD'].rolling(window=rolling_window).mean()
    data['GOLD_roll_std_5'] = data['GOLD'].rolling(window=rolling_window).std()
    data['GOLD_roll_min_5'] = data['GOLD'].rolling(window=rolling_window).min()
    data['GOLD_roll_max_5'] = data['GOLD'].rolling(window=rolling_window).max()

    # custom features (same-day data, shifted one row)
    data['Gold_Oil_Ratio'] = data['GOLD'] / data['CrudeOil_SpotPrice_BrentUK']
    data['USD_CNY_to_JPY'] = data['SpotRate_USD_CNY'] / data['SpotRate_Tokyo_9AM_USD_JPY']
    data['China_US_CPI_Ratio'] = data['China_CPI_YoY_CurrentMonth'] / data['US_CPI_YoY_NSA']

    # rate of change (same-day data, shifted one row)
    data['Gold_Rate_of_Change'] = data['GOLD'].pct_change(periods=5)
    data['Oil_Rate_of_Change'] = data['CrudeOil_SpotPrice_BrentUK'].pct_change(periods=5)

    # trend (same-day data, shifted one row)
    data['Gold_Trend_7_days'] = data['GOLD'].rolling(window=7).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )

    # inflation-adjusted gold price (same-day data, shifted one row)
    data['Gold_Inflation_Adjusted'] = data['GOLD'] / data['US_CPI_YoY_NSA']

    # volatility (same-day data, shifted one row)
    data['Gold_Volatility_10_days'] = data['GOLD'].rolling(window=10).std()
    data['Oil_Volatility_10_days'] = data['CrudeOil_SpotPrice_BrentUK'].rolling(window=10).std()

    # interactions (same-day data, shifted one row)
    data['Gold_SP500_Ratio'] = data['GOLD'] / data['US_SP500_Index']
    data['US_Japan_Interest_Rate_Diff'] = data['US_DowJones_IndustrialAverage'] - data['SpotRate_Tokyo_9AM_USD_JPY']

    # time features
    data['Month'] = data['Date'].dt.month
    data['Quarter'] = data['Date'].dt.quarter
    data['Day_of_Week'] = data['Date'].dt.dayofweek

    # shift the same-day features
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

    # check all features exist
    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        raise ValueError(f"The following required features are missing from the data: {missing_features}")

    # split X and y
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

# FinancialDataset, preprocess_data etc. are assumed to be defined elsewhere
# adjust these to your own needs

class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CNNLSTM, self).__init__()

        # conv layers
        # 48 features per sample, so input_size=1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1)  # input_size=1
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        # LSTM
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=3, batch_first=True, dropout=0.3)

        # dropout
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x is (batch_size, seq_len)
        print(f"Input shape: {x.shape}")  # debug: print shape

        # one feature per sample, add a channel dim
        x = x.unsqueeze(1)  # now (batch_size, 1, seq_len)
        print(f"Shape after unsqueeze: {x.shape}")  # debug: print shape

        # conv layers
        x = self.conv1(x)
        x = nn.ReLU()(x)
        print(f"Shape after conv1: {x.shape}")  # debug: print shape

        x = self.conv2(x)
        x = nn.ReLU()(x)
        print(f"Shape after conv2: {x.shape}")  # debug: print shape

        x = self.conv3(x)
        x = nn.ReLU()(x)
        print(f"Shape after conv3: {x.shape}")  # debug: print shape

        # Conv1d gives (batch_size, out_channels, seq_len); transpose to (batch_size, seq_len, out_channels)
        x = x.transpose(1, 2)  # -> (batch_size, seq_len, 256)
        print(f"Shape before LSTM: {x.shape}")  # debug: print shape

        # LSTM
        x, (h_n, c_n) = self.lstm(x)  # input is now (batch_size, seq_len, 256)
        print(f"Shape after LSTM: {x.shape}")  # debug: print shape

        # take the last time step of the LSTM
        x = x[:, -1, :]  # (batch_size, hidden_size)
        print(f"Shape after extracting last LSTM output: {x.shape}")  # debug: print shape

        # dropout
        x = self.dropout(x)

        # return the last time step
        return x



def main():
    # file path
    file_path = "/home/w/桌面/lilj/GOLD_cleaned.xlsx"
    try:
        X, y = preprocess_data(file_path)
    except Exception as e:
        print(f"Error in preprocessing data: {e}")
        logging.error(f"Error in preprocessing data: {e}")
        return

    # class distribution
    print("Class Distribution:")
    print(y.value_counts())
    logging.info(f"Class Distribution:\n{y.value_counts()}")

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logging.info(f"Using device: {device}")

    # time series split
    tscv = TimeSeriesSplit(n_splits=5)

    # metric lists
    accuracies, precisions, recalls, f1_scores, aucs = [], [], [], [], []

    # first CV: train the CNN-LSTM
    print("Training CNN-LSTM model...")
    logging.info("Training CNN-LSTM model started.")

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"Fold {fold + 1} - Training CNN-LSTM")
        logging.info(f"Fold {fold + 1} - Training CNN-LSTM started")

        # split
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # datasets and loaders
        train_dataset = FinancialDataset(X_train_scaled, y_train)
        test_dataset = FinancialDataset(X_test_scaled, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # model
        model = CNNLSTM(input_size=X_train_scaled.shape[1], hidden_size=128, num_classes=2).to(device)

        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

        # training
        try:
            model.train()
            for epoch in range(100):  # 10 epochs
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)

                    # forward
                    optimizer.zero_grad()
                    output = model(data)

                    # loss + backward
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

            print(f"Fold {fold + 1}: Training completed.")
        except Exception as e:
            print(f"Error during CNN-LSTM training: {e}")
            logging.error(f"Error during CNN-LSTM training: {e}")
            return

        # evaluation
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

        # metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # metric lists
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        aucs.append(auc)

        # print and log metrics
        print(
            f"Fold {fold + 1}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
        logging.info(
            f"Fold {fold + 1}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    # overall results
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
