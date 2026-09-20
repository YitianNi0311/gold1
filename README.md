# BFNE-Net experiment settings

## Random seeds

Three seeds: **42, 43, 44**.

- Neural nets and numpy/random: `seed + fold`
- Optuna (TPE sampler): `seed * 10000 + fold`
- `random_state` of the final tree models: the seed itself
- The LightGBM inside the Optuna objective always uses `random_state=42`

## Data and splits

- `GOLD_cleaned.xlsx`, 4,419 rows after cleaning, 48 features
- `TimeSeriesSplit(n_splits=5)`, 736 rows per test fold
- Classification label: `GOLD.shift(-1) > GOLD` (next-day direction); regression label: `GOLD.shift(-1)` (next-day price)
- Features are standardized; the regression target is standardized with a scaler fit on each training fold and inverted back to raw prices for evaluation

## Evaluation protocol

- Base models for fold k are trained on the data before fold k and predict fold k (out-of-fold predictions).
- The meta learner that scores fold k is trained only on the out-of-fold predictions of folds 1 to k-1, so it never sees a label from the block it predicts.
- Fold 1 therefore only trains the first meta learner; every model in the tables is scored on folds 2-5 (4 x 736 = 2,944 predictions per seed).
- The Optuna search for LightGBM uses a 3-split `TimeSeriesSplit` inside each training fold.

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

## CNN-LSTM

Conv1d 128-256-512 (kernel 3) -> 3-layer LSTM (hidden 128, dropout 0.3) -> Linear. Batch size 32, up to 100 epochs, 30-epoch inner tuning (patience 8) over learning rates 5e-4 / 1e-3 / 2e-4 (AdamW, weight decay 1e-5) plus Adam 5e-4, CosineAnnealing (T_max=100).

## Random walk

Predicts the next trading day's GOLD with today's GOLD. No parameters.

## Statistical test

Diebold-Mariano, full BFNE-Net vs the version without FCNNs: two-sided, horizon 1, Bartlett-weighted HAC variance with lag 10, Harvey-Leybourne-Newbold small-sample correction.
