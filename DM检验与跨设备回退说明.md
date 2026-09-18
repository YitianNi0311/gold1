# DM 检验和跨设备回退

`diebold_mariano.py` 比较完整 `归一.py` 模型与真正的零 FCNN 模型。它要求每个模型在相同的 `fold`、`date`、`row_index` 上各有一条预测，并逐行核对真实分类标签、原始 GOLD 价格及保存的 3,680 条历史测试目标。五折必须各有 736 条。

预先指定的逐日损失是原价平方误差、原价绝对误差、百分数绝对误差、分类 Brier 损失与分类 0–1 损失。F1、AUC 和 RMSE 不是逐日可加损失，不直接代入 DM。损失差定义为“完整模型 − 零 FCNN”；正值表示零 FCNN 损失较小。使用双侧检验、一步预测 horizon=1、Bartlett 权重 HAC 长期方差、固定滞后 10 个交易观测和 Harvey–Leybourne–Newbold 小样本修正。

三随机种子的 DM 表先把 42、43、44 三次运行在**同一日期**的两模型损失各自平均，然后对 3,680 个日期的平均配对损失差检验。三次运行的 p 值不会被直接平均。每个种子的逐日损失另存，方便复核。

这批历史分类目标仍是 `(GOLD.diff() > 0)`，它表示当日相对前日的方向；元模型也见过评估区间的标签。检验代码中的 horizon=1 是损失序列的小样本修正设置，不能把分类结果解释成真正的下一日方向预测。DM 的 p 值只能描述旧协议中保存的误差序列，不能当作无泄漏、前瞻性下一日预测能力的证明。新表须把此限制与重建基线来源写明。

已在远端保存回退标记 `pre-dm-tables-20260918`，对应提交 `6a4705ac6a0bcec1c70ebffb43ededdebbef01d0`。在另一台电脑查看改动前版本：

```shell
git clone https://github.com/YitianNi0311/gold1.git
cd gold1
git fetch --tags
git switch -c before-dm pre-dm-tables-20260918
```

若以后要撤销已发布的新代码或结果，应在 `main` 上按**结果提交、代码提交**的逆序执行 `git revert` 并推送，保留完整提交历史。旧表格文件不会被新实验覆盖；六张表会另行提交，便于单独撤销。

方法依据：[Diebold 与 Mariano (1995)](https://doi.org/10.1080/07350015.1995.10524599)、[Harvey、Leybourne 与 Newbold (1997)](https://doi.org/10.1016/S0169-2070(96)00719-4)。
