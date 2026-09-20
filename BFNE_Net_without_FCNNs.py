"""BFNE-Net without the FCNNs: LightGBM + XGBoost + Random Forest -> GBM.

Features, labels, scaling and search settings are the same as the original; only the
tree learners are used here."""

import logging

import numpy as np
import optuna
import pandas as pd
import ta
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBClassifier, XGBRegressor


def preprocess_data(file_path):
    # load data
    data = pd.read_excel(file_path)

    # make Date a datetime
    if not np.issubdtype(data['Date'].dtype, np.datetime64):
        data['Date'] = pd.to_datetime(data['Date'])

    # old label: is today's gold price up vs yesterday
    data['Next_Day_Change'] = (data['GOLD'].diff() > 0).astype(int)

    # regression target: next day's gold price
    data['Next_Day_Price'] = data['GOLD'].shift(-1)

    # feature engineering
    # lags
    for lag in range(1, 31):
        data[f'GOLD_lag{lag}'] = data['GOLD'].shift(lag)

    # moving averages
    data['GOLD_MA_3_days'] = data['GOLD'].rolling(window=3).mean()
    data['GOLD_MA_5_days'] = data['GOLD'].rolling(window=5).mean()
    data['GOLD_MA_10_days'] = data['GOLD'].rolling(window=10).mean()

    # technical indicators
    data['RSI'] = ta.momentum.RSIIndicator(close=data['GOLD'], window=14).rsi()
    macd = ta.trend.MACD(close=data['GOLD'])
    data['MACD'] = macd.macd_diff()

    # bollinger bands
    bollinger = ta.volatility.BollingerBands(close=data['GOLD'], window=20)
    data['Bollinger_High'] = bollinger.bollinger_hband()
    data['Bollinger_Low'] = bollinger.bollinger_lband()

    # rolling stats
    rolling_window = 10
    data['GOLD_roll_mean_5'] = data['GOLD'].rolling(window=rolling_window).mean()
    data['GOLD_roll_std_5'] = data['GOLD'].rolling(window=rolling_window).std()
    data['GOLD_roll_min_5'] = data['GOLD'].rolling(window=rolling_window).min()
    data['GOLD_roll_max_5'] = data['GOLD'].rolling(window=rolling_window).max()

    # custom features
    data['Gold_Oil_Ratio'] = data['GOLD'] / data['CrudeOil_SpotPrice_BrentUK']
    data['USD_CNY_to_JPY'] = data['SpotRate_USD_CNY'] / data['SpotRate_Tokyo_9AM_USD_JPY']
    data['China_US_CPI_Ratio'] = data['China_CPI_YoY_CurrentMonth'] / data['US_CPI_YoY_NSA']

    # rate of change
    data['Gold_Rate_of_Change'] = data['GOLD'].pct_change(periods=5)
    data['Oil_Rate_of_Change'] = data['CrudeOil_SpotPrice_BrentUK'].pct_change(periods=5)

    # trend
    data['Gold_Trend_7_days'] = data['GOLD'].rolling(window=7).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )

    # inflation-adjusted gold price
    data['Gold_Inflation_Adjusted'] = data['GOLD'] / data['US_CPI_YoY_NSA']

    # volatility
    data['Gold_Volatility_10_days'] = data['GOLD'].rolling(window=10).std()
    data['Oil_Volatility_10_days'] = data['CrudeOil_SpotPrice_BrentUK'].rolling(window=10).std()

    # interactions
    data['Gold_SP500_Ratio'] = data['GOLD'] / data['US_SP500_Index']
    data['US_Japan_Interest_Rate_Diff'] = data['US_DowJones_IndustrialAverage'] - data['SpotRate_Tokyo_9AM_USD_JPY']

    # time features
    data['Month'] = data['Date'].dt.month
    data['Quarter'] = data['Date'].dt.quarter
    data['Day_of_Week'] = data['Date'].dt.dayofweek

    # shift same-day features back one day
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

    # make sure every feature exists
    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        raise ValueError(f"The following required features are missing from the data: {missing_features}")

    # split X and y
    X = data[features]
    y_class = data['Next_Day_Change']
    y_reg = data['Next_Day_Price']

    return X, y_class, y_reg


def preprocess_and_scale(X_train, X_test, scaler_type='StandardScaler'):
    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_type == 'RobustScaler':
        scaler = RobustScaler()
    else:
        raise ValueError("Unsupported scaler type. Choose 'StandardScaler' or 'RobustScaler'.")

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def preprocess_and_scale_reg_target(y_train, y_test, scaler_type='StandardScaler'):
    if scaler_type == 'StandardScaler':
        scaler_y = StandardScaler()
    elif scaler_type == 'RobustScaler':
        scaler_y = RobustScaler()
    else:
        raise ValueError("Unsupported scaler type. Choose 'StandardScaler' or 'RobustScaler'.")

    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
    return y_train_scaled, y_test_scaled, scaler_y


def objective_lgbm_reg(trial, X_train, y_train):
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

    lgbm = LGBMRegressor(**param)
    score = cross_val_score(lgbm, X_train, y_train, cv=3, scoring='neg_mean_squared_error').mean()
    return -score


def optimize_lgbm_reg(X_train, y_train):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_lgbm_reg(trial, X_train, y_train), n_trials=50)

    print("Best parameters for LGBMRegressor:", study.best_params)
    print("Best MSE for LGBMRegressor:", study.best_value)
    logging.info(f"Best parameters for LGBMRegressor: {study.best_params}")
    logging.info(f"Best MSE for LGBMRegressor: {study.best_value}")

    return study.best_params


def objective_lgbm_clf(trial, X_train, y_train):
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


def optimize_lgbm_clf(X_train, y_train):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_lgbm_clf(trial, X_train, y_train), n_trials=50)

    print("Best parameters for LGBMClassifier:", study.best_params)
    print("Best ROC AUC for LGBMClassifier:", study.best_value)
    logging.info(f"Best parameters for LGBMClassifier: {study.best_params}")
    logging.info(f"Best ROC AUC for LGBMClassifier: {study.best_value}")

    return study.best_params


def build_regressors(X_train, y_train):
    best_params = optimize_lgbm_reg(X_train, y_train)
    return (
        LGBMRegressor(**best_params, random_state=42, n_jobs=-1),
        XGBRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=7,
            subsample=0.9, colsample_bytree=0.9,
            objective="reg:squarederror", random_state=42, n_jobs=-1,
        ),
        RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1,
        ),
    )


def build_classifiers(X_train, y_train):
    best_params = optimize_lgbm_clf(X_train, y_train)
    return (
        LGBMClassifier(**best_params, random_state=42, n_jobs=-1),
        XGBClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=7,
            subsample=0.9, colsample_bytree=0.9,
            objective="binary:logistic", use_label_encoder=False,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        ),
        RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1,
        ),
    )


def prepare_fold(X, y_class, y_reg, train_index, test_index):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    yr_train, yr_test = y_reg.iloc[train_index], y_reg.iloc[test_index]
    Xr_train, Xr_test, _ = preprocess_and_scale(X_train, X_test)
    Xc_train, Xc_test, _ = preprocess_and_scale(X_train, X_test)
    yr_train_fit, yr_test_fit, scaler_y = preprocess_and_scale_reg_target(yr_train, yr_test)
    return {
        "Xr_train": Xr_train, "Xr_test": Xr_test,
        "Xc_train": Xc_train, "Xc_test": Xc_test,
        "yc_train": y_class.iloc[train_index], "yc_test": y_class.iloc[test_index],
        "yr_train": yr_train, "yr_test": yr_test,
        "yr_train_fit": yr_train_fit, "yr_test_fit": yr_test_fit,
        "scaler_y": scaler_y,
    }


def train_fold(fold_data):
    regressors = build_regressors(fold_data["Xr_train"], fold_data["yr_train_fit"])
    for model in regressors:
        model.fit(fold_data["Xr_train"], fold_data["yr_train_fit"])
    regression_features = np.column_stack([
        model.predict(fold_data["Xr_test"]) for model in regressors
    ])

    classifiers = build_classifiers(fold_data["Xc_train"], fold_data["yc_train"].to_numpy())
    for model in classifiers:
        model.fit(fold_data["Xc_train"], fold_data["yc_train"].to_numpy())
    classification_features = np.hstack([
        model.predict_proba(fold_data["Xc_test"]) for model in classifiers
    ])
    return regression_features, classification_features


def fit_meta_models(regression_features, regression_labels, classification_features, classification_labels):
    regressor = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.15, max_depth=5, random_state=42,
    )
    classifier = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.15, max_depth=5, random_state=42,
    )
    regressor.fit(np.vstack(regression_features), np.asarray(regression_labels))
    classifier.fit(np.vstack(classification_features), np.asarray(classification_labels))
    return regressor, classifier
