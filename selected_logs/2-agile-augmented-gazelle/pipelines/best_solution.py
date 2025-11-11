import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import lightgbm as lgb
import os

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

# Parse datetime and create cyclical features plus continuous time trend
train["c1"] = pd.to_datetime(train["c1"])
test["c1"] = pd.to_datetime(test["c1"])
# Compute the reference start time from training set
min_time = train["c1"].min()

for df in [train, test]:
    # datetime parts
    df["year"] = df["c1"].dt.year
    df["month"] = df["c1"].dt.month
    df["day"] = df["c1"].dt.day
    df["hour"] = df["c1"].dt.hour
    df["weekday"] = df["c1"].dt.weekday
    # cyclical encodings
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    # continuous trend feature: seconds since first training timestamp
    df["elapsed"] = (df["c1"] - min_time).dt.total_seconds()

# Features and target
base_feats = [f"c{i}" for i in range(2, 10)]
time_feats = [
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]
features = base_feats + time_feats + ["elapsed"]
target = "c12"

X = train[features]
y = train[target].apply(np.log1p)
X_test = test[features]

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_list = []

for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model = lgb.LGBMRegressor(objective="regression", random_state=42, n_estimators=100)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)

    preds[preds < 0] = 0
    rmse = mean_squared_error(y_val, preds, squared=False)
    rmse_list.append(rmse)

mean_rmse = np.mean(rmse_list)
print(f"CV mean RMSE: {mean_rmse:.4f}")

# Train on full data and predict test
final_model = lgb.LGBMRegressor(
    objective="regression", random_state=42, n_estimators=100
)
final_model.fit(X, y)
test_preds = final_model.predict(X_test)
test_preds[test_preds < 0] = 0
# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"c1": test["c1"].dt.strftime("%Y-%m-%d %H:%M:%S"), "c12": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
