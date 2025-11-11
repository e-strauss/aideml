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

# Identify target
target_col = sample.columns[1]  # 'count'

# Feature engineering
for df in [train, test]:
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df["weekofyear"] = df["datetime"].dt.isocalendar().week.astype(int)
    # Cyclical encoding
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
    # Interaction features
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
]

X = train[features]
y = train[target_col].values
y_log = np.log1p(y)

# 5-fold CV with early stopping via callbacks
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []
best_iters = []
for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y_log[train_idx], y_log[val_idx]
    model = LGBMRegressor(
        learning_rate=0.05, num_leaves=31, n_estimators=5000, n_jobs=-1, random_state=42
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
    rmsle = np.sqrt(
        mean_squared_error(np.log1p(train[target_col].values[val_idx]), np.log1p(preds))
    )
    rmsle_scores.append(rmsle)

cv_rmsle = np.mean(rmsle_scores)
mean_best_iter = int(np.mean(best_iters))
print(f"CV RMSLE: {cv_rmsle:.5f}, Avg Best Iter: {mean_best_iter}")

# Train final model with averaged best iteration
final_model = LGBMRegressor(
    learning_rate=0.05,
    num_leaves=31,
    n_estimators=mean_best_iter,
    n_jobs=-1,
    random_state=42,
)
final_model.fit(X, y_log)

# Predict on test
test_X = test[features]
test_preds_log = final_model.predict(test_X)
test_preds = np.expm1(test_preds_log)
test_preds[test_preds < 0] = 0

# Save submission
os.makedirs("./working", exist_ok=True)
submission = sample.copy()
submission[target_col] = np.round(test_preds).astype(int)
submission.to_csv("./working/submission.csv", index=False)
