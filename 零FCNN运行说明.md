# 本地零 FCNN 模型

模型代码为 `no_fcnn_model.py`，运行入口为 `run_no_fcnn.py`，全部位于当前工作区。

当前入口按 `归一.py` 的重复评估次数运行：一次五折 OOF、五轮五折回归评估；分类只在第一轮五折计分。`零FCNN运行逻辑核对.md` 记录五轮改动前的旧入口对照。Optuna sampler 仍未固定自身种子，不能保证正式训练预测逐数值相同。

结构：**LightGBM + XGBoost + Random Forest → GBM**。分类和回归分别训练三种基模型及一个 GBM 元模型，回归元特征 3 维、分类概率元特征 6 维。代码不导入 PyTorch、不创建神经网络，也不加载历史实验运行器。

当前工作区所有 Python 代码均不包含统计检验实现：不计算 p 值、显著性或重采样置信区间。输出仅包括模型预测、常规分类/回归指标、五折均值与标准差、数据划分和运行日志。五折标准差是描述性汇总。

## 运行

在当前工作区 PowerShell 中执行：

```powershell
& 'C:/Users/WKU-MATH-558/miniconda3/envs/nyt/python.exe' run_no_fcnn.py
```

默认读取脚本旁的 `GOLD_cleaned.xlsx`，输出到 `no_fcnn_results/run_时间戳/`。也可指定数据和一个尚不存在的输出目录：

```powershell
& 'C:/Users/WKU-MATH-558/miniconda3/envs/nyt/python.exe' run_no_fcnn.py --data GOLD_cleaned.xlsx --output no_fcnn_results/my_run
```

## 保留的实验设置

- 预处理、48 个特征、标签、特征/目标缩放、Optuna 搜索函数直接按当前 `归一.py` 的定义实现，自动化检查对比了这 7 个函数的 AST，确认一致。
- 分类标签仍是 `(GOLD.diff() > 0).astype(int)`；回归标签仍是 `GOLD.shift(-1)`。
- `TimeSeriesSplit(n_splits=5)`；先在五折产生 OOF 元特征并训练 GBM，再在相同五折进行五轮重新训练和评估。元模型按原脚本顺序累计拟合六次。
- LightGBM 每个任务、每折、每次训练均执行原 50 trials，目标内部 `cv=3`；完整运行共 3,000 trials。每次搜索预算保持不变，重复轮次增加了总训练量。
- XGBoost：300 trees、lr=0.03、depth=7、行列采样=0.9。
- Random Forest：200 trees、depth=10；`max_features` 和其余未显式设置项保持原库默认值。
- GBM：300 stages、lr=0.15、depth=5；模型 `random_state=42`。
- Optuna 沿用当前 `归一.py` 的默认 sampler，未额外注入 sampler seed。因此不能保证不同次搜索得到完全相同的最优参数。

历史标签与输入存在泄漏；元模型训练也已见过最终评估区间的标签。这些问题按用户要求保留，并写入控制台、日志和 `run_manifest.json`；不能把所得历史协议成绩称为无泄漏的下一日预测成绩。

## 回归尺度和文件

本模型为与完整 stacking 模型保持一致，回归目标在每个训练折标准化。最终元模型预测通过该折训练得到的 y scaler 逆变换一次，以真实原始 GOLD 价格计算 RMSE（USD/oz）和 MAPE（%）。这与三个独立树模型直接使用原价目标训练的方式不同。

输出目录包含：

- `training_log.txt`、`run_manifest.json`：日志、完成/失败状态、配置和历史协议说明。
- `fold1_predictions.csv` 至 `fold5_predictions.csv`、`predictions.csv`：第一轮的分类和回归逐日预测，共 3,680 条。
- `regression_predictions_all_rounds.csv`、`round{轮次}_fold{折}_regression.csv`：五轮回归逐日预测，共 25 个折记录、18,400 行；重复日期带轮次标识。
- `round{轮次}_scaler_y_fold{折}.json`：每轮每折 y scaler 的均值和尺度。
- `fold_metrics.csv`：25 个回归折指标，其中第一轮五折还含分类指标。
- `metric_summary.csv`：回归 25 折、分类 5 折的均值及总体标准差（ddof=0）。
- `pooled_metrics.json`：只合并第一轮 3,680 条样本，避免重复日期及与折均值混淆。
- `fold_boundaries.csv`：训练和测试日期边界。

自动化代码检查可直接在本地运行，不需要历史实验目录：

```powershell
& 'C:/Users/WKU-MATH-558/miniconda3/envs/nyt/python.exe' -m unittest -v test_regression_scale test_additional_scales test_no_fcnn test_reconstructed_alignment
```

本次验证包含小型模拟数据上的五轮评估调用顺序、真实 GBM 元模型拟合、三个基模型的构造参数、尺度恢复和 CSV 输出；耗时的基模型搜索使用模拟替身。尚未对 GOLD 数据启动正式训练。
