import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error
)
from regression_scale_utils import regression_metrics_original_price
import warnings
import ta  # technical indicators lib
from pathlib import Path

warnings.filterwarnings('ignore')  # silence warnings

# preprocessing
def preprocess_data(file_path):
    data = pd.read_excel(file_path)

    if not np.issubdtype(data['Date'].dtype, np.datetime64):
        data['Date'] = pd.to_datetime(data['Date'])

    # classification and regression targets
    data['Next_Day_Change'] = (data['GOLD'].shift(-1) > data['GOLD']).astype(int)
    data['Next_Day_Gold_Price'] = data['GOLD'].shift(-1)

    # feature engineering
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

    # rolling features
    data['GOLD_roll_mean_5'] = data['GOLD'].rolling(window=10).mean()
    data['GOLD_roll_std_5'] = data['GOLD'].rolling(window=10).std()
    data['GOLD_roll_min_5'] = data['GOLD'].rolling(window=10).min()
    data['GOLD_roll_max_5'] = data['GOLD'].rolling(window=10).max()

    # custom features
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

    # time features
    data['Month'] = data['Date'].dt.month
    data['Quarter'] = data['Date'].dt.quarter
    data['Day_of_Week'] = data['Date'].dt.dayofweek

    # shift features by one day
    today_features = [
        'Gold_Oil_Ratio', 'USD_CNY_to_JPY', 'China_US_CPI_Ratio',
        'Gold_Rate_of_Change', 'Oil_Rate_of_Change', 'Gold_Trend_7_days',
        'Gold_Inflation_Adjusted', 'Gold_Volatility_10_days', 'Oil_Volatility_10_days',
        'Gold_SP500_Ratio', 'US_Japan_Interest_Rate_Diff'
    ]
    data[today_features] = data[today_features].shift(1)

    # drop NaN / inf rows
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    # original feature set
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


def train_random_forest(X_train, y_cls_train, y_reg_train, X_test, y_cls_test, y_reg_test):
    # classifier
    rf_cls = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_cls.fit(X_train, y_cls_train)

    y_cls_pred = rf_cls.predict(X_test)
    y_cls_prob = rf_cls.predict_proba(X_test)[:, 1]

    # regressor
    rf_reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_reg.fit(X_train, y_reg_train)

    y_reg_pred = rf_reg.predict(X_test)

    # classification metrics
    acc = accuracy_score(y_cls_test, y_cls_pred)
    prec = precision_score(y_cls_test, y_cls_pred)
    rec = recall_score(y_cls_test, y_cls_pred)
    f1 = f1_score(y_cls_test, y_cls_pred)
    auc = roc_auc_score(y_cls_test, y_cls_prob)

    # regression metrics
    # y was never standardized, so score the price directly
    _, rmse, mape = regression_metrics_original_price(y_reg_test, y_reg_pred)

    return acc, prec, rec, f1, auc, rmse, mape, y_reg_pred


def main():
    file_path = Path(__file__).resolve().parent / 'GOLD_cleaned.xlsx'
    X, y_cls, y_reg = preprocess_data(file_path)

    tscv = TimeSeriesSplit(n_splits=5)

    results = {
        "accuracy": [], "precision": [], "recall": [], "f1": [], "auc": [],
        "rmse_usd_per_oz": [], "mape_percent": []
    }
    regression_predictions = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"Fold {fold + 1}")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_cls_train, y_cls_test = y_cls.iloc[train_index], y_cls.iloc[test_index]
        y_reg_train, y_reg_test = y_reg.iloc[train_index], y_reg.iloc[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        acc, prec, rec, f1, auc, rmse, mape, y_reg_pred = train_random_forest(
            X_train_scaled, y_cls_train, y_reg_train, X_test_scaled, y_cls_test, y_reg_test
        )

        results["accuracy"].append(acc)
        results["precision"].append(prec)
        results["recall"].append(rec)
        results["f1"].append(f1)
        results["auc"].append(auc)
        results["rmse_usd_per_oz"].append(rmse)
        results["mape_percent"].append(mape)
        regression_predictions.append(pd.DataFrame({
            "fold": fold + 1,
            "row_index": X_test.index.to_numpy(),
            "y_true_price_original": np.asarray(y_reg_test),
            "y_pred_price_original": np.asarray(y_reg_pred),
        }))

        print(f"Fold {fold + 1} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        print(f"Fold {fold + 1} - RMSE (USD/oz): {rmse:.4f}, MAPE (%): {mape:.4f}")

    print("\nOverall Results:")
    labels = {"rmse_usd_per_oz": "RMSE (USD/oz)", "mape_percent": "MAPE (%)"}
    for metric in results:
        print(f"{labels.get(metric, metric.capitalize())}: {np.mean(results[metric]):.4f} ± {np.std(results[metric]):.4f}")
    pd.concat(regression_predictions, ignore_index=True).to_csv(
        "RF_regression_predictions_original_price.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "metric": [labels.get(metric, metric) for metric in results],
        "mean": [np.mean(results[metric]) for metric in results],
        "std": [np.std(results[metric]) for metric in results],
    }).to_csv("RF_metric_summary.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
