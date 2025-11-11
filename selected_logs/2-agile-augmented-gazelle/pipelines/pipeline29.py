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
min_datetime = train["datetime"].min()
for df in [train, test]:
    df["elapsed_time"] = (df["datetime"] - min_datetime).dt.total_seconds() / 3600.0
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

X = train[features].reset_index(drop=True)
y = train[target_col].values
y_log = np.log1p(y)
test_X = test[features].reset_index(drop=True)

# prepare keys for encoding
train_keys = train["season"].astype(str) + "_" + train["hour"].astype(str)
test_keys = test["season"].astype(str) + "_" + test["hour"].astype(str)

# 5-fold CV with target encoding per fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []
best_iters = []
season_hour_te = np.zeros(len(train))

for train_idx, val_idx in kf.split(X):
    # compute mapping on training fold
    keys_tr = train_keys.iloc[train_idx]
    ytr = y_log[train_idx]
    df_tr = pd.DataFrame({"key": keys_tr, "y": ytr})
    mapping = df_tr.groupby("key")["y"].mean().to_dict()
    global_mean = ytr.mean()
    # map to validation
    keys_val = train_keys.iloc[val_idx]
    te_val = keys_val.map(mapping).fillna(global_mean).values
    season_hour_te[val_idx] = te_val

    # extend features
    X_tr_ext = X.iloc[train_idx].copy()
    X_val_ext = X.iloc[val_idx].copy()
    X_tr_ext["season_hour_te"] = keys_tr.map(mapping).fillna(global_mean).values
    X_val_ext["season_hour_te"] = te_val

    # fit model
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
        X_tr_ext,
        y_log[train_idx],
        eval_set=[(X_val_ext, y_log[val_idx])],
        eval_metric="rmse",
        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=0)],
    )
    best_iters.append(model.best_iteration_)
    preds_log = model.predict(X_val_ext, num_iteration=model.best_iteration_)
    preds = np.expm1(preds_log)
    preds[preds < 0] = 0
    rmsle = np.sqrt(
        mean_squared_error(np.log1p(train[target_col].values[val_idx]), np.log1p(preds))
    )
    rmsle_scores.append(rmsle)

cv_rmsle = np.mean(rmsle_scores)
mean_best_iter = int(np.mean(best_iters))
print(f"CV RMSLE with season_hour_te: {cv_rmsle:.5f}, Avg Best Iter: {mean_best_iter}")

# add season_hour_te to full train and test
df_full = pd.DataFrame({"key": train_keys, "y": y_log})
full_map = df_full.groupby("key")["y"].mean().to_dict()
full_mean = y_log.mean()
train_te_full = train_keys.map(full_map).fillna(full_mean).values
test_te = test_keys.map(full_map).fillna(full_mean).values

X_full = X.copy()
X_full["season_hour_te"] = train_te_full
test_full = test_X.copy()
test_full["season_hour_te"] = test_te

# final model training
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

# predict on test
preds_log = final_model.predict(test_full)
preds = np.expm1(preds_log)
preds[preds < 0] = 0

# save submission
os.makedirs("./working", exist_ok=True)
submission = sample.copy()
submission[target_col] = np.round(preds).astype(int)
submission.to_csv("./working/submission.csv", index=False)
