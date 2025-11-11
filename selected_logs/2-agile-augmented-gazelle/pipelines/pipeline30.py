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
min_datetime = train["datetime"].min()
for df in [train, test]:
    df["elapsed_time"] = (df["datetime"] - min_datetime).dt.total_seconds() / 3600.0

# Feature engineering
for df in [train, test]:
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df["weekofyear"] = df["datetime"].dt.isocalendar().week.astype(int)
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
    df["hour_workingday"] = df["hour"] * df["workingday"]
    df["hour_holiday"] = df["hour"] * df["holiday"]

# Define static features (hour_te will be added dynamically)
base_features = [
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

X = train[base_features].copy()
y = train[target_col].values
y_log = np.log1p(y)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []
best_iters = []

for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_tr, y_val = y_log[train_idx], y_log[val_idx]
    # K-fold target encoding for hour
    y_tr_series = pd.Series(y_tr, index=X_tr.index)
    hour_map = y_tr_series.groupby(X_tr["hour"]).mean().to_dict()
    global_hour_mean = y_tr_series.mean()
    X_tr["hour_te"] = X_tr["hour"].map(hour_map).fillna(global_hour_mean)
    X_val["hour_te"] = X_val["hour"].map(hour_map).fillna(global_hour_mean)
    # Model
    model = LGBMRegressor(
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=5000,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
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
    rmsle = np.sqrt(mean_squared_error(y_val, preds_log))
    rmsle_scores.append(rmsle)

cv_rmsle = np.mean(rmsle_scores)
mean_best_iter = int(np.mean(best_iters))
print(f"CV RMSLE: {cv_rmsle:.5f}, Avg Best Iter: {mean_best_iter}")

# Full training with hour_te
# Prepare full train encoding
full_map = pd.Series(y_log, index=X.index).groupby(train["hour"]).mean().to_dict()
full_global = y_log.mean()
X_full = X.copy()
X_full["hour_te"] = X_full["hour"].map(full_map).fillna(full_global)

final_model = LGBMRegressor(
    learning_rate=0.05,
    num_leaves=31,
    n_estimators=mean_best_iter,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
)
final_model.fit(X_full, y_log)

# Prepare test
test_X = test[base_features].copy()
test_X["hour_te"] = test_X["hour"].map(full_map).fillna(full_global)
preds_log_test = final_model.predict(test_X)
preds_test = np.expm1(preds_log_test).clip(0, None)

# Save submission
os.makedirs("./working", exist_ok=True)
submission = sample.copy()
submission[target_col] = np.round(preds_test).astype(int)
submission.to_csv("./working/submission.csv", index=False)
