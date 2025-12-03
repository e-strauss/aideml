import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
sample = pd.read_csv("./input/sampleSubmission.csv")


# Feature engineering
def add_time_features(df):
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    return df


train = add_time_features(train)
test = add_time_features(test)

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

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []
for tr_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
    )
    model.fit(X_tr, y_tr)
    pred_val = model.predict(X_val)
    score = np.sqrt(mean_squared_error(y_val, pred_val))
    rmsle_scores.append(score)

mean_rmsle = np.mean(rmsle_scores)
print(f"Mean RMSLE: {mean_rmsle:.5f}")

# Retrain on full data and predict test set
final_model = xgb.XGBRegressor(
    n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
)
final_model.fit(X, y)
pred_test = final_model.predict(X_test)
pred_test = np.expm1(pred_test)
pred_test[pred_test < 0] = 0

# Create submission
sample["count"] = pred_test.astype(int)
sample.to_csv("./working/submission.csv", index=False)
