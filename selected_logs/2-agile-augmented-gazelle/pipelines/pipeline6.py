import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

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

X = train[features]
y = train[target_col].values
y_log = np.log1p(y)

# 5-fold CV with LightGBM
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []
best_iters = []
for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y_log[train_idx], y_log[val_idx]
    model = lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.05,
        n_estimators=1000,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        early_stopping_rounds=50,
        verbose=False,
    )
    best_iters.append(model.best_iteration_)
    preds_log = model.predict(X_val, num_iteration=model.best_iteration_)
    preds = np.expm1(preds_log)
    preds[preds < 0] = 0
    rmsle = np.sqrt(mean_squared_error(np.log1p(np.expm1(y_val)), np.log1p(preds)))
    rmsle_scores.append(rmsle)

cv_rmsle = np.mean(rmsle_scores)
print(f"CV RMSLE: {cv_rmsle:.5f}")

# Train final model
final_n_estimators = int(np.median(best_iters))
final_model = lgb.LGBMRegressor(
    objective="regression",
    learning_rate=0.05,
    n_estimators=final_n_estimators,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
)
final_model.fit(X, y_log, verbose=False)

# Predict on test set
test_X = test[features]
test_preds_log = final_model.predict(test_X)
test_preds = np.expm1(test_preds_log)
test_preds[test_preds < 0] = 0

# Save submission
submission = sample.copy()
submission[target_col] = test_preds.astype(int)
submission.to_csv("./working/submission.csv", index=False)
