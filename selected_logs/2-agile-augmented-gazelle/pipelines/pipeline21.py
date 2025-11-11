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

# Basic datetime feature engineering
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

# Core features
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

X = train[base_features].copy()
y = train[target_col].values
y_log = np.log1p(y)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []

for train_idx, val_idx in kf.split(X):
    X_tr = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    y_tr_log = y_log[train_idx]
    y_val = y[val_idx]
    # K-fold TE for (hour,dayofweek)
    df_map = pd.DataFrame(
        {"hour": X_tr["hour"], "dayofweek": X_tr["dayofweek"], "y_log": y_tr_log}
    )
    grp_mean = df_map.groupby(["hour", "dayofweek"])["y_log"].mean()
    mapping = grp_mean.to_dict()
    global_mean = y_tr_log.mean()
    # Map to train
    X_tr["grp"] = list(zip(X_tr["hour"], X_tr["dayofweek"]))
    X_tr["hour_dow_te"] = X_tr["grp"].map(mapping).fillna(global_mean)
    X_tr.drop(columns="grp", inplace=True)
    # Map to val
    X_val["grp"] = list(zip(X_val["hour"], X_val["dayofweek"]))
    X_val["hour_dow_te"] = X_val["grp"].map(mapping).fillna(global_mean)
    X_val.drop(columns="grp", inplace=True)
    # Train model
    features = base_features + ["hour_dow_te"]
    model = LGBMRegressor(
        learning_rate=0.05, num_leaves=31, n_estimators=1000, n_jobs=-1, random_state=42
    )
    model.fit(X_tr[features], y_tr_log)
    preds_log = model.predict(X_val[features])
    preds = np.expm1(preds_log)
    preds[preds < 0] = 0
    rmsle = np.sqrt(mean_squared_error(np.log1p(y_val), np.log1p(preds)))
    rmsle_scores.append(rmsle)

cv_rmsle = np.mean(rmsle_scores)
print(f"CV RMSLE: {cv_rmsle:.5f}")

# Final mapping on full data for test
df_map_full = pd.DataFrame(
    {"hour": X["hour"], "dayofweek": X["dayofweek"], "y_log": y_log}
)
grp_mean_full = df_map_full.groupby(["hour", "dayofweek"])["y_log"].mean()
mapping_full = grp_mean_full.to_dict()
global_mean_full = y_log.mean()

test_X = test[base_features].copy()
test_X["grp"] = list(zip(test_X["hour"], test_X["dayofweek"]))
test_X["hour_dow_te"] = test_X["grp"].map(mapping_full).fillna(global_mean_full)
test_X.drop(columns="grp", inplace=True)

# Train on full and predict
final_model = LGBMRegressor(
    learning_rate=0.05, num_leaves=31, n_estimators=1000, n_jobs=-1, random_state=42
)
final_model.fit(
    pd.concat([X, test_X[base_features + ["hour_dow_te"]]])[: len(X)], y_log
)
test_preds_log = final_model.predict(test_X[base_features + ["hour_dow_te"]])
test_preds = np.expm1(test_preds_log)
test_preds[test_preds < 0] = 0

# Save submission
os.makedirs("./working", exist_ok=True)
submission = sample.copy()
submission[target_col] = np.round(test_preds).astype(int)
submission.to_csv("./working/submission.csv", index=False)
