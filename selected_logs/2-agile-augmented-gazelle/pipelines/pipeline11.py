import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
import warnings
import os

warnings.filterwarnings("ignore")

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
sample = pd.read_csv("./input/sampleSubmission.csv")


# Feature engineering
def fe(df):
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    return df


train = fe(train)
test = fe(test)

# Define features and target
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
    "hour",
    "weekday",
]
X = train[features]
y = train["count"]
X_test = test[features]


# RMSLE function
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.clip(y_pred, 0, None)))


# 5-fold CV with GradientBoostingRegressor
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    y_tr_log = np.log1p(y_tr)
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr_log)
    preds_log = model.predict(X_val)
    preds = np.expm1(preds_log)
    scores.append(rmsle(y_val, preds))

print(f"CV RMSLE: {np.mean(scores):.5f}")

# Train on full data and predict
model_full = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_full.fit(X, np.log1p(y))
preds_test = np.expm1(model_full.predict(X_test))
preds_test = np.clip(preds_test, 0, None)

# Save submission
os.makedirs("./working", exist_ok=True)
sample["count"] = preds_test
sample.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
