import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import ta
import logging
from sklearn.preprocessing import RobustScaler
# Update the file path
file_path = "/home/w/桌面/lilj/GOLD_cleaned.xlsx"


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configure logging
logging.basicConfig(
    filename='training_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Define Dataset class
class FinancialDataset(torch.utils.data.Dataset):
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


# Define CNN-LSTM Model
class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2, kernel_size=3, dropout=0.5):
        super(CNNLSTMModel, self).__init__()

        # CNN part
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # LSTM part
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, batch_first=True, bidirectional=False)

        # Fully connected part
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # CNN forward pass
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        # Prepare input for LSTM (LSTM expects a 3D tensor)
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, feature_dim)

        # LSTM forward pass
        x, _ = self.lstm(x)

        # Only use the output of the last time step for classification
        x = x[:, -1, :]

        # Fully connected layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# Data Preprocessing (same as provided)
def preprocess_data(file_path):
    # Load data
    data = pd.read_excel(file_path)

    # Ensure 'Date' is datetime
    if not np.issubdtype(data['Date'].dtype, np.datetime64):
        data['Date'] = pd.to_datetime(data['Date'])

    # Create target variable
    data['Next_Day_Change'] = (data['GOLD'].diff() > 0).astype(int)

    # Feature Engineering
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

    # Custom Features (shifted)
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

    data['Month'] = data['Date'].dt.month
    data['Quarter'] = data['Date'].dt.quarter
    data['Day_of_Week'] = data['Date'].dt.dayofweek

    today_features = [
        'Gold_Oil_Ratio', 'USD_CNY_to_JPY', 'China_US_CPI_Ratio',
        'Gold_Rate_of_Change', 'Oil_Rate_of_Change', 'Gold_Trend_7_days',
        'Gold_Inflation_Adjusted', 'Gold_Volatility_10_days', 'Oil_Volatility_10_days',
        'Gold_SP500_Ratio', 'US_Japan_Interest_Rate_Diff'
    ]
    data[today_features] = data[today_features].shift(1)

    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    features = [
        'GOLD_MA_3_days', 'GOLD_lag1', 'GOLD_MA_5_days',
        'GOLD_lag2', 'GOLD_MA_10_days', 'GOLD_lag3', 'GOLD_lag4', 'GOLD_lag5',
        'GOLD_lag6', 'GOLD_lag7', 'GOLD_lag8', 'GOLD_lag9', 'GOLD_lag10',
        'Gold_Oil_Ratio', 'USD_CNY_to_JPY', 'China_US_CPI_Ratio',
        'Gold_Rate_of_Change', 'Oil_Rate_of_Change', 'Gold_Trend_7_days',
        'Gold_Inflation_Adjusted', 'Gold_Volatility_10_days', 'Oil_Volatility_10_days',
        'Gold_SP500_Ratio', 'US_Japan_Interest_Rate_Diff', 'RSI', 'MACD',
        'Bollinger_High', 'Bollinger_Low', 'GOLD_roll_mean_5', 'GOLD_roll_std_5',
        'GOLD_roll_min_5', 'GOLD_roll_max_5', 'Month', 'Quarter', 'Day_of_Week'
    ]

    data = data[features + ['Next_Day_Change']]

    X = data[features]
    y = data['Next_Day_Change']

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# Prepare DataLoader
def prepare_dataloader(X, y, batch_size=64):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Train Model
def train_model(model, train_loader, val_loader, epochs=50, patience=10, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=patience, delta=0)

    model.to(device)
    best_model = None
    best_val_auc = -np.inf

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        model.eval()
        val_preds = []
        val_probs = []
        val_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                val_preds.extend(preds)
                val_probs.extend(probs)
                val_labels.extend(labels.cpu().numpy())

        # Metrics
        acc = accuracy_score(val_labels, val_preds)
        prec = precision_score(val_labels, val_preds)
        rec = recall_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds)
        auc = roc_auc_score(val_labels, val_probs)

        logging.info(f"Epoch [{epoch + 1}/{epochs}] - Loss: {running_loss / len(train_loader):.4f} - "
                     f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        # Log best model
        if auc > best_val_auc:
            best_val_auc = auc
            best_model = model.state_dict()
            # Save the best model checkpoint
            torch.save(best_model, 'best_model.pth')

        # Early stopping
        early_stopping(auc)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered.")
            break

    model.load_state_dict(best_model)
    return model

# Main Execution
if __name__ == "__main__":
    # Preprocess the data
    X, y = preprocess_data('path_to_your_data.xlsx')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Prepare data loaders
    train_loader = prepare_dataloader(X_train, y_train)
    test_loader = prepare_dataloader(X_test, y_test)

    # Initialize model
    input_dim = X_train.shape[1]  # Number of features
    model = CNNLSTMModel(input_dim=input_dim, hidden_dim=64, num_classes=2)

    # Train the model
    trained_model = train_model(model, train_loader, test_loader, epochs=50, patience=10)

    # Evaluate the model on test data
    model.eval()
    test_preds = []
    test_probs = []
    test_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = trained_model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            test_preds.extend(preds)
            test_probs.extend(probs)
            test_labels.extend(labels.cpu().numpy())

    # Metrics
    acc = accuracy_score(test_labels, test_preds)
    prec = precision_score(test_labels, test_preds)
    rec = recall_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds)
    auc = roc_auc_score(test_labels, test_probs)

    print(f"Test Results - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
