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

# Basic feature engineering
for df in [train, test]:
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df["weekofyear"] = df["datetime"].dt.isocalendar().week.astype(int)
    # Cyclical
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
    # Interactions
    df["hour_workingday"] = df["hour"] * df["workingday"]
    df["hour_holiday"] = df["hour"] * df["holiday"]

# Prepare target
y = train[target_col].values
y_log = np.log1p(y)

# Create K-fold target encoding for month
kf_enc = KFold(n_splits=5, shuffle=True, random_state=42)
month_te = np.zeros(train.shape[0], dtype=float)
for tr_idx, val_idx in kf_enc.split(train):
    # Compute mean log target by month on train fold
    mapping = pd.Series(y_log[tr_idx]).groupby(train["month"].iloc[tr_idx]).mean()
    # Map to validation
    month_te[val_idx] = train["month"].iloc[val_idx].map(mapping).values
# For test set, use full-train mapping
full_mapping = pd.Series(y_log).groupby(train["month"]).mean()
test_month_te = test["month"].map(full_mapping).values

# Add encoded feature
train["month_te"] = month_te
test["month_te"] = test_month_te

# Feature list
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
    "month_te",
]

X = train[features]
X_test = test[features]

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []
for tr_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y_log[tr_idx], y[val_idx]
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

# Train full model and predict
final_model = LGBMRegressor(
    learning_rate=0.05, num_leaves=31, n_estimators=1000, n_jobs=-1, random_state=42
)
final_model.fit(X, y_log)
preds_test_log = final_model.predict(X_test)
preds_test = np.expm1(preds_test_log)
preds_test[preds_test < 0] = 0

# Save submission
os.makedirs("./working", exist_ok=True)
submission = sample.copy()
submission[target_col] = np.round(preds_test).astype(int)
submission.to_csv("./working/submission.csv", index=False)
