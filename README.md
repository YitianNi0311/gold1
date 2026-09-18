# Gold prediction experiments

黄金价格回归、历史方向分类和 stacking 模型的代码与数据存档。

## 当前版本

- `归一.py`：FCNN1 + FCNN2 + LightGBM + XGBoost + Random Forest → GBM，保留原有重复评估流程。
- `no_fcnn_model.py`、`run_no_fcnn.py`：独立零 FCNN 模型，LightGBM + XGBoost + Random Forest → GBM。一次五折 OOF、五轮五折回归评估；分类只在第一轮五折计分。
- `reconstructed_random_walk.py`、`reconstructed_cnn_lstm.py`、`reconstructed_without_fe.py`：历史空白项的独立重建代码。它们不是已找到的作者原实现；当前未写入 Table 8/9 成绩。
- `RF.py`、`XGBoost.py`、`gold lightgbm.py`：独立树模型。
- `GOLD_cleaned.xlsx`：当前实验工作簿。
- `regression_scale_utils.py`：统一原始价格 RMSE（USD/oz）和百分数 MAPE。
- `历史状态复核_20260917/`：先前保存预测的复核指标和证据。

## 运行

使用已安装 numpy、pandas、scikit-learn、lightgbm、xgboost、optuna、ta 和 openpyxl 的 Python 环境：

```shell
python run_no_fcnn.py
python -m unittest -v test_regression_scale test_no_fcnn
```

完整 FCNN 模型及其他脚本还需要其各自导入的 PyTorch、matplotlib 等依赖。部分历史脚本保留原机器的数据路径，运行前需检查。

## 历史协议说明

分类标签为 `(GOLD.diff() > 0)`，回归标签为 `GOLD.shift(-1)`。旧分类标签和元模型评估流程存在泄漏，保存的历史结果不能解释为无泄漏的下一日预测成绩。

本工作区不包含统计检验实现。自动化代码测试用于验证程序和评估尺度。

详见 [代码修复与重建说明](代码修复与重建说明_20260918.md)、[零 FCNN 运行说明](零FCNN运行说明.md)和[回归尺度修改说明](回归尺度修改说明.md)。旧的[运行逻辑核对](零FCNN运行逻辑核对.md)记录五轮评估改动前的实现。历史复核文档中引用的外部目录不随本仓库上传；对齐所需的真实目标已单独保存在 `seed42_reference_targets.csv`。
