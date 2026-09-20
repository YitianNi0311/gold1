# BFNE-Net experiment settings

## Random seeds

Three seeds: **42, 43, 44**.

- Neural nets and numpy/random: `seed + 100 * evaluation_round + fold`
- Optuna (TPE sampler): `seed * 10000 + fold_slot`
- `random_state` of the final tree models: the seed itself
- The LightGBM inside the Optuna objective always uses `random_state=42`
- Only the first evaluation round is used, five folds

## Data and splits

- `GOLD_cleaned.xlsx`, 4,419 rows after cleaning, 48 features
- `TimeSeriesSplit(n_splits=5)`, 736 test rows per fold, 3,680 test predictions in total
- Classification label: `GOLD.diff() > 0`; regression label: `GOLD.shift(-1)`
- Features are standardized; the regression target is standardized with a scaler fit on each training fold and inverted back to raw prices for evaluation

## Neural networks (FCNN1 / FCNN2)

| Parameter | FCNN1 | FCNN2 |
|---|---|---|
| Hidden layers | 256-128-64 | 512-256-128-64 |
| Activation / norm / dropout | LeakyReLU / BatchNorm / 0.5 | same |
| Optimizer | AdamW, weight decay 1e-5 | same |
| Learning rate | regression 5e-3, classification 5e-4 | same |
| LR schedule | CosineAnnealing, T_max=50 | same |
| Batch size | 32 | 32 |
| Max epochs | 300 | 300 |
| Early-stopping patience | regression 12, classification 15 (both nets stop when either triggers) | same |
| Loss | regression MSE; classification focal loss (gamma=2, balanced class weights) | same |

## Tree models

| Model | Parameters |
|---|---|
| Random Forest | n_estimators=200, max_depth=10 |
| XGBoost | n_estimators=300, learning_rate=0.03, max_depth=7, subsample=0.9, colsample_bytree=0.9; regression reg:squarederror, classification binary:logistic (logloss) |
| LightGBM (inside BFNE-Net) | Optuna, 50 trials, 3-fold CV; n_estimators 100-500, learning_rate 0.01-0.1 (log), num_leaves 20-100, subsample 0.5-1.0, colsample_bytree 0.5-1.0, max_depth 3-15; regression minimizes MSE, classification maximizes AUC |
| LightGBM (standalone baseline) | n_estimators=500, learning_rate=0.01, max_depth=15, num_leaves=100, subsample=0.5, colsample_bytree=0.5 |

## Meta learner (GBM)

GradientBoosting, n_estimators=300, learning_rate=0.15, max_depth=5.
Inputs: full BFNE-Net uses 5 features for regression and 10 for classification; the version without FCNNs uses 3 and 6.
The meta learner is fit on the five-fold OOF predictions first, then evaluated on the same five folds.

## CNN-LSTM

Conv1d 128-256-512 (kernel 3) -> 3-layer LSTM (hidden 128, dropout 0.3) -> Linear. Batch size 32, up to 100 epochs, 30-epoch inner tuning (patience 8) over learning rates 5e-4 / 1e-3 / 2e-4 (AdamW, weight decay 1e-5) plus Adam 5e-4, CosineAnnealing (T_max=100).

## Random walk

Predicts the next trading day's GOLD with today's GOLD. No parameters.

## Statistical test

Diebold-Mariano, full BFNE-Net vs the version without FCNNs: two-sided, horizon 1, Bartlett-weighted HAC variance with lag 10, Harvey-Leybourne-Newbold small-sample correction.
