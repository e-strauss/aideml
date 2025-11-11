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
target_col = sample.columns[1]  # 'count'

# Feature engineering
for df in [train, test]:
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

# Prepare target
y = train[target_col].values
y_log = np.log1p(y)
train["y_log"] = y_log

# Base features
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
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

# 5-fold CV with target encoding on 'hour'
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []

for train_idx, val_idx in kf.split(train):
    tr_df = train.iloc[train_idx].copy()
    val_df = train.iloc[val_idx].copy()
    # compute target encoding on log scale
    mapping = tr_df.groupby("hour")["y_log"].mean()
    global_mean = tr_df["y_log"].mean()
    # map to train and val
    tr_df["hour_te"] = tr_df["hour"].map(mapping).fillna(global_mean)
    val_df["hour_te"] = val_df["hour"].map(mapping).fillna(global_mean)
    X_tr = tr_df[features + ["hour_te"]]
    X_val = val_df[features + ["hour_te"]]
    y_tr = tr_df["y_log"].values
    y_val = val_df[target_col].values

    model = LGBMRegressor(
        learning_rate=0.05, num_leaves=31, n_estimators=1000, n_jobs=-1, random_state=42
    )
    model.fit(X_tr, y_tr)
    preds_log = model.predict(X_val)
    preds = np.expm1(preds_log)
    preds[preds < 0] = 0
    rmsle = np.sqrt(mean_squared_error(np.log1p(y_val), np.log1p(preds)))
    rmsle_scores.append(rmsle)

cv_rmsle = np.mean(rmsle_scores)
print(f"CV RMSLE: {cv_rmsle:.5f}")

# Final model training with full data target encoding
mapping_full = train.groupby("hour")["y_log"].mean()
global_mean_full = train["y_log"].mean()
train["hour_te"] = train["hour"].map(mapping_full).fillna(global_mean_full)
test["hour_te"] = test["hour"].map(mapping_full).fillna(global_mean_full)

X_full = train[features + ["hour_te"]]
test_X = test[features + ["hour_te"]]

final_model = LGBMRegressor(
    learning_rate=0.05, num_leaves=31, n_estimators=1000, n_jobs=-1, random_state=42
)
final_model.fit(X_full, train["y_log"].values)

# Predict test
test_preds_log = final_model.predict(test_X)
test_preds = np.expm1(test_preds_log)
test_preds[test_preds < 0] = 0

# Prepare submission
os.makedirs("./working", exist_ok=True)
submission = sample.copy()
submission[target_col] = np.round(test_preds).astype(int)
submission.to_csv("./working/submission.csv", index=False)
