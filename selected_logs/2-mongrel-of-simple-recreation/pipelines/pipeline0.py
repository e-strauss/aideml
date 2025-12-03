import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")


# Feature engineering
def add_datetime_features(df):
    dt = pd.to_datetime(df["datetime"])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.weekday
    return df


train = add_datetime_features(train)
test = add_datetime_features(test)

features = [
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
]
X = train[features]
y = np.log1p(train["count"])
X_test = test[features]

# Cross‐validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []

for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X_tr, y_tr)
    pred_log = model.predict(X_val)
    score = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(pred_log)))
    rmsle_scores.append(score)

cv_score = np.mean(rmsle_scores)
print(f"CV RMSLE: {cv_score:.5f}")

# Train on full data and predict
final_model = lgb.LGBMRegressor(random_state=42)
final_model.fit(X, y)
pred_log_test = final_model.predict(X_test)
pred_count = np.expm1(pred_log_test).clip(0)

# Prepare submission
submission = pd.DataFrame({"datetime": test["datetime"], "count": pred_count})
submission.to_csv("./working/submission.csv", index=False)
