import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import os

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
sample = pd.read_csv("./input/sampleSubmission.csv")
target_col = sample.columns[1]

# Feature engineering
for df in [train, test]:
    df["datetime"] = pd.to_datetime(df["datetime"])
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
test_X = test[features]

# Bagging ensemble parameters
n_bags = 5
seeds = [42 + 10 * i for i in range(n_bags)]

# 5-fold CV with bagging
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []

for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y_log[train_idx], y[val_idx]
    val_preds = np.zeros(len(X_val))

    for seed in seeds:
        model = LGBMRegressor(
            learning_rate=0.05,
            num_leaves=31,
            n_estimators=1000,
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(X_tr, y_tr)
        preds_log = model.predict(X_val)
        preds = np.expm1(preds_log)
        val_preds += preds

    val_preds /= n_bags
    val_preds[val_preds < 0] = 0
    rmsle = np.sqrt(mean_squared_error(np.log1p(y_val), np.log1p(val_preds)))
    rmsle_scores.append(rmsle)

cv_rmsle = np.mean(rmsle_scores)
print(f"CV RMSLE (bagged): {cv_rmsle:.5f}")

# Train final bagged models and predict test
test_preds = np.zeros(len(test_X))
for seed in seeds:
    model = LGBMRegressor(
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=1000,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X, y_log)
    preds = np.expm1(model.predict(test_X))
    test_preds += preds

test_preds /= n_bags
test_preds[test_preds < 0] = 0

# Save submission
os.makedirs("./working", exist_ok=True)
submission = sample.copy()
submission[target_col] = np.round(test_preds).astype(int)
submission.to_csv("./working/submission.csv", index=False)
