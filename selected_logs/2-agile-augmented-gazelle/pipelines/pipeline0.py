import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

# Load data
train = pd.read_csv("./input/train.csv", parse_dates=["datetime"])
test = pd.read_csv("./input/test.csv", parse_dates=["datetime"])


# Feature engineering
def add_datetime_features(df):
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    return df


train = add_datetime_features(train)
test = add_datetime_features(test)

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
    "dayofweek",
    "hour",
]
X = train[features]
y = train["count"]


# RMSLE scorer
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.clip(y_pred, 0, None)))


# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    scores.append(rmsle(y_val, preds))

print(f"5-fold RMSLE: {np.mean(scores):.5f}")

# Train on full data and predict test
final_model = lgb.LGBMRegressor(random_state=42)
final_model.fit(X, y)
test_preds = final_model.predict(test[features])
test_preds = np.clip(test_preds, 0, None).round().astype(int)

# Save submission
submission = pd.DataFrame(
    {"datetime": test["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"), "count": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
