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
]

X_base = train[base_features].copy()
y = train[target_col].values
y_log = np.log1p(y)
test_base = test[base_features].copy()

# 5‐fold CV with K‐fold target encoding for 'season'
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []

for train_idx, val_idx in kf.split(X_base):
    X_tr = X_base.iloc[train_idx].copy()
    X_val = X_base.iloc[val_idx].copy()
    y_tr_log = y_log[train_idx]

    # compute mapping from season to mean y_log in fold
    mapping = pd.Series(y_tr_log, index=X_tr.index).groupby(X_tr["season"]).mean()
    global_mean = y_tr_log.mean()

    # add encoded feature
    X_tr["season_te"] = X_tr["season"].map(mapping).fillna(global_mean)
    X_val["season_te"] = X_val["season"].map(mapping).fillna(global_mean)

    # train model
    model = LGBMRegressor(
        learning_rate=0.05, num_leaves=31, n_estimators=1000, n_jobs=-1, random_state=42
    )
    model.fit(X_tr, y_tr_log)

    # predict and evaluate
    preds_log = model.predict(X_val)
    preds = np.expm1(preds_log)
    preds[preds < 0] = 0
    rmsle = np.sqrt(mean_squared_error(np.log1p(y[val_idx]), np.log1p(preds)))
    rmsle_scores.append(rmsle)

cv_rmsle = np.mean(rmsle_scores)
print(f"CV RMSLE with season_te: {cv_rmsle:.5f}")

# Train final model on full data including season_te
mapping_full = pd.Series(y_log, index=X_base.index).groupby(X_base["season"]).mean()
global_mean_full = y_log.mean()
X_full = X_base.copy()
X_full["season_te"] = X_full["season"].map(mapping_full).fillna(global_mean_full)
test_full = test_base.copy()
test_full["season_te"] = test_full["season"].map(mapping_full).fillna(global_mean_full)

final_model = LGBMRegressor(
    learning_rate=0.05, num_leaves=31, n_estimators=1000, n_jobs=-1, random_state=42
)
final_model.fit(X_full, y_log)
test_preds_log = final_model.predict(test_full)
test_preds = np.expm1(test_preds_log)
test_preds[test_preds < 0] = 0

# Save submission
os.makedirs("./working", exist_ok=True)
submission = sample.copy()
submission[target_col] = np.round(test_preds).astype(int)
submission.to_csv("./working/submission.csv", index=False)
