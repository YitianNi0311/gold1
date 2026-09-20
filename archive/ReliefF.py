import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import ta
from skrebate import ReliefF

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
         'SaudiArabia_CPI_YoY', 'Turkey_CPI_YoY',
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

    # 重置索引，确保索引一致
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X, y, features

# 数据路径
file_path = '/home/w/桌面/lilj/GOLD_cleaned.xlsx'

# 调用预处理函数
X, y, features = preprocess_data(file_path)

# 确保没有缺失值
X = X.dropna(axis=0)  # 删除含有缺失值的行
y = y[X.index]  # 确保 y 与 X 对齐

# 检查 X 是否是 DataFrame，并确保列名为字符串
if not isinstance(X, pd.DataFrame):
    raise ValueError("X should be a pandas DataFrame")
if not all(isinstance(col, str) for col in X.columns):
    raise ValueError("X columns should be strings")

# 将 X 转换为 numpy 数组
X = X.values

# 创建 ReliefF 模型
relieff = ReliefF(n_neighbors=20, n_features_to_select=48)

# 训练模型以计算特征重要性
relieff.fit(X, y)

# 获取特征重要性
importances = relieff.feature_importances_

# 排序数据：按照特征重要性从大到小排序
sorted_idx = np.argsort(importances)[::-1]
sorted_features = np.array(features)[sorted_idx]  # 使用原始的特征名称
sorted_importances = np.array(importances)[sorted_idx]

# 绘制特征重要性图
plt.figure(figsize=(10, 8))

# 使用条形图
bars = plt.barh(sorted_features, sorted_importances, color=mcolors.LinearSegmentedColormap.from_list("blue_orange", ["#2D6CAB", "#F17D56"])(np.linspace(0, 1, len(sorted_features))))

# 添加标题和标签
plt.title('Feature Importance (ReliefF)', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Features', fontsize=14)

# 反转y轴，使得最重要的特征在顶部
plt.gca().invert_yaxis()

# 增加条形之间的间距控制
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.12, right=0.88)  # 控制上下左右的空隙

# 保存图像而不是显示
plt.tight_layout(pad=3.0)  # 确保布局没有重叠，增加间距
plt.savefig('feature_importance_relieff.png')  # 保存为 PNG 文件
plt.close()  # 关闭图形，避免在一些环境中显示出来
