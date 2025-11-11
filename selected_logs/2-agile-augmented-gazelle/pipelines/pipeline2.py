import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
sample = pd.read_csv("./input/sampleSubmission.csv")

# Identify target column from sample submission
target_col = sample.columns[1]  # 'count'

# Feature engineering
for df in [train, test]:
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour

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
]

X = train[features]
y = train[target_col].values

# Log-transform target
y_log = np.log1p(y)

# 5-fold CV for RMSLE
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []
for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y_log[train_idx], y[val_idx]
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    preds_log = model.predict(X_val)
    preds = np.expm1(preds_log)
    preds[preds < 0] = 0
    rmsle = np.sqrt(mean_squared_error(np.log1p(y_val), np.log1p(preds)))
    rmsle_scores.append(rmsle)

cv_rmsle = np.mean(rmsle_scores)
print(f"CV RMSLE: {cv_rmsle:.5f}")

# Train on full data and predict test
final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
final_model.fit(X, y_log)
test_preds_log = final_model.predict(test[features])
test_preds = np.expm1(test_preds_log)
test_preds[test_preds < 0] = 0

# Prepare submission
submission = sample.copy()
submission[target_col] = test_preds.astype(int)
submission.to_csv("./working/submission.csv", index=False)
