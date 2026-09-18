import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ta


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
        'CrudeOil_SpotPrice_BrentUK', 'China_CPI_YoY_CurrentMonth', 'US_CPI_YoY_NSA',
        'Japan_CPI_YoY', 'SaudiArabia_CPI_YoY', 'Turkey_CPI_YoY',
        'US_DowJones_IndustrialAverage', 'US_SP500_Index', 'SpotRate_USD_CNY',
        'Interbank_OpenRate_USD_INR', 'SpotRate_Tokyo_9AM_USD_JPY'
    ]

    # 检查特征是否齐全
    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        raise ValueError(f"The following required features are missing from the data: {missing_features}")

    # 划分特征和目标变量
    X = data[features]
    y = data['Next_Day_Change']

    return X, y

# 读取和预处理数据
file_path = "/home/w/桌面/lilj/GOLD_cleaned.xlsx"
X, y = preprocess_data(file_path)

# 计算相关性矩阵（使用Spearman方法）
correlation_matrix = X.corr(method='spearman')

# 设置图形大小
plt.figure(figsize=(20, 15))

# 绘制相关性矩阵的热图
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar=True)

# 设置标题
plt.title("Spearman Correlation Matrix of Features")

# 保存输出图像
output_path = "feature_HeatMap_spearman.png"
plt.savefig(output_path)


