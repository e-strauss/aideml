import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
import os

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
sample = pd.read_csv("./input/sampleSubmission.csv")
target_col = sample.columns[1]  # 'count'

# Parse datetime
train["datetime"] = pd.to_datetime(train["datetime"])
test["datetime"] = pd.to_datetime(test["datetime"])

# Compute elapsed time
min_dt = train["datetime"].min()
for df in (train, test):
    df["elapsed_time"] = (df["datetime"] - min_dt).dt.total_seconds() / 3600.0

# Feature engineering
for df in (train, test):
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df["weekofyear"] = df["datetime"].dt.isocalendar().week.astype(int)
    # cyclical
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * (df["dayofyear"] - 1) / 365)
    df["doy_cos"] = np.cos(2 * np.pi * (df["dayofyear"] - 1) / 365)
    df["woy_sin"] = np.sin(2 * np.pi * (df["weekofyear"] - 1) / 52)
    df["woy_cos"] = np.cos(2 * np.pi * (df["weekofyear"] - 1) / 52)
    # interactions
    df["hour_workingday"] = df["hour"] * df["workingday"]
    df["hour_holiday"] = df["hour"] * df["holiday"]

features = [
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "year",
    "month",
    "dayofweek",
    "hour",
    "dayofyear",
    "weekofyear",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "woy_sin",
    "woy_cos",
    "hour_workingday",
    "hour_holiday",
    "elapsed_time",
]

X = train[features]
y = train[target_col].values
y_log = np.log1p(y)
test_X = test[features]

# Ensemble settings
seeds = [42, 52, 62]
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []
best_iters = []

# 5-fold CV with ensemble
for tr_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y_log[tr_idx], y_log[val_idx]
    preds_sum = np.zeros(len(val_idx))
    for seed in seeds:
        model = LGBMRegressor(
            learning_rate=0.05,
            num_leaves=31,
            n_estimators=5000,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=0)],
        )
        best_iters.append(model.best_iteration_)
        preds_log = model.predict(X_val, num_iteration=model.best_iteration_)
        preds = np.expm1(preds_log)
        preds[preds < 0] = 0
        preds_sum += preds
    preds_avg = preds_sum / len(seeds)
    rmsle = np.sqrt(mean_squared_error(np.log1p(y[val_idx]), np.log1p(preds_avg)))
    rmsle_scores.append(rmsle)

cv_rmsle = np.mean(rmsle_scores)
mean_best_iter = int(np.mean(best_iters))
print(f"CV Ensemble RMSLE: {cv_rmsle:.5f}, Avg Best Iter: {mean_best_iter}")

# Train final ensemble on full data
test_preds_sum = np.zeros(len(test_X))
for seed in seeds:
    model = LGBMRegressor(
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=mean_best_iter,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X, y_log)
    preds_log = model.predict(test_X)
    preds = np.expm1(preds_log)
    preds[preds < 0] = 0
    test_preds_sum += preds
test_preds = test_preds_sum / len(seeds)

# Save submission
os.makedirs("./working", exist_ok=True)
submission = sample.copy()
submission[target_col] = np.round(test_preds).astype(int)
submission.to_csv("./working/submission.csv", index=False)
