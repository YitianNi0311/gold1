import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_percentage_error
)
from torch.utils.data import DataLoader, TensorDataset
import warnings
import ta

warnings.filterwarnings('ignore')  # Suppress warnings

# 数据预处理函数
def preprocess_data(file_path):
    data = pd.read_excel(file_path)

    if not np.issubdtype(data['Date'].dtype, np.datetime64):
        data['Date'] = pd.to_datetime(data['Date'])

    data['Next_Day_Change'] = (data['GOLD'].diff() > 0).astype(int)
    data['Next_Day_Gold_Price'] = data['GOLD'].shift(-1)

    for lag in range(1, 31):
        data[f'GOLD_lag{lag}'] = data['GOLD'].shift(lag)

    data['GOLD_MA_3_days'] = data['GOLD'].rolling(window=3).mean()
    data['GOLD_MA_5_days'] = data['GOLD'].rolling(window=5).mean()
    data['GOLD_MA_10_days'] = data['GOLD'].rolling(window=10).mean()

    data['RSI'] = ta.momentum.RSIIndicator(close=data['GOLD'], window=14).rsi()
    macd = ta.trend.MACD(close=data['GOLD'])
    data['MACD'] = macd.macd_diff()

    data['Gold_Rate_of_Change'] = data['GOLD'].pct_change(periods=5)

    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    features = [col for col in data.columns if col.startswith("GOLD") or col in ['RSI', 'MACD']]
    X = data[features]
    y_cls = data['Next_Day_Change']
    y_reg = data['Next_Day_Gold_Price']

    return X, y_cls, y_reg

# 定义更复杂的FCNN模型
class FCNN(nn.Module):
    def __init__(self, input_dim):
        super(FCNN, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.hidden4 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.regression_head = nn.Linear(64, 1)
        self.classification_head = nn.Linear(64, 2)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        reg_output = self.regression_head(x)
        cls_output = self.classification_head(x)
        return reg_output, cls_output

def train_fcnn(X_train, y_cls_train, y_reg_train, X_test, y_cls_test, y_reg_test):
    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FCNN(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_cls_train_tensor = torch.tensor(y_cls_train.values, dtype=torch.long).to(device)
    y_reg_train_tensor = torch.tensor(y_reg_train.values, dtype=torch.float32).to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_cls_test_tensor = torch.tensor(y_cls_test.values, dtype=torch.long).to(device)
    y_reg_test_tensor = torch.tensor(y_reg_test.values, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_cls_train_tensor, y_reg_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(50):
        for batch_X, batch_y_cls, batch_y_reg in train_loader:
            optimizer.zero_grad()
            reg_output, cls_output = model(batch_X)
            loss_cls = criterion_cls(cls_output, batch_y_cls)
            loss_reg = criterion_reg(reg_output.squeeze(), batch_y_reg)
            loss = loss_cls + loss_reg
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        reg_output, cls_output = model(X_test_tensor)
        cls_preds = torch.argmax(cls_output, axis=1).cpu().numpy()
        reg_preds = reg_output.cpu().numpy()

    acc = accuracy_score(y_cls_test, cls_preds)
    prec = precision_score(y_cls_test, cls_preds)
    rec = recall_score(y_cls_test, cls_preds)
    f1 = f1_score(y_cls_test, cls_preds)
    auc = roc_auc_score(y_cls_test, torch.softmax(cls_output, axis=1)[:, 1].cpu().numpy())

    rmse = np.sqrt(mean_squared_error(y_reg_test, reg_preds))
    mape = mean_absolute_percentage_error(y_reg_test, reg_preds) * 100.0

    return acc, prec, rec, f1, auc, rmse, mape

def main():
    file_path = '/home/w/桌面/lilj/GOLD_cleaned.xlsx'
    X, y_cls, y_reg = preprocess_data(file_path)

    tscv = TimeSeriesSplit(n_splits=5)

    results = {
        "accuracy": [], "precision": [], "recall": [], "f1": [], "auc": [], "rmse": [], "mape": []
    }

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"Fold {fold + 1}")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_cls_train, y_cls_test = y_cls.iloc[train_index], y_cls.iloc[test_index]
        y_reg_train, y_reg_test = y_reg.iloc[train_index], y_reg.iloc[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        acc, prec, rec, f1, auc, rmse, mape = train_fcnn(
            X_train_scaled, y_cls_train, y_reg_train, X_test_scaled, y_cls_test, y_reg_test
        )

        results["accuracy"].append(acc)
        results["precision"].append(prec)
        results["recall"].append(rec)
        results["f1"].append(f1)
        results["auc"].append(auc)
        results["rmse"].append(rmse)
        results["mape"].append(mape)

        print(f"Fold {fold + 1} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    print("\nOverall Results:")
    print(f"Average Accuracy: {np.mean(results['accuracy']):.4f} ± {np.std(results['accuracy']):.4f}")
    print(f"Average Precision: {np.mean(results['precision']):.4f} ± {np.std(results['precision']):.4f}")
    print(f"Average Recall: {np.mean(results['recall']):.4f} ± {np.std(results['recall']):.4f}")
    print(f"Average F1: {np.mean(results['f1']):.4f} ± {np.std(results['f1']):.4f}")
    print(f"Average AUC: {np.mean(results['auc']):.4f} ± {np.std(results['auc']):.4f}")
    print(f"Average RMSE (USD/oz): {np.mean(results['rmse']):.4f} ± {np.std(results['rmse']):.4f}")
    print(f"Average MAPE (%): {np.mean(results['mape']):.4f} ± {np.std(results['mape']):.4f}")

if __name__ == "__main__":
    main()
