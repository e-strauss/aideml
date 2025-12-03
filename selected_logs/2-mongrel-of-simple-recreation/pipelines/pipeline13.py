import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

# Load data
train = pd.read_csv("./input/train.csv", parse_dates=["datetime"])
test = pd.read_csv("./input/test.csv", parse_dates=["datetime"])
sample = pd.read_csv("./input/sampleSubmission.csv")


# Feature engineering
def add_date_features(df):
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    return df


train = add_date_features(train)
test = add_date_features(test)

features = [
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "holiday",
    "workingday",
    "season",
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
rmsles = []

for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_val)
    rmsle = np.sqrt(mean_squared_error(y_val, y_pred))
    rmsles.append(rmsle)

print(f"Mean RMSLE: {np.mean(rmsles):.5f}")

# Train final model
final_model = HistGradientBoostingRegressor(random_state=42)
final_model.fit(X, y)

# Predict on test
preds = final_model.predict(X_test)
preds = np.expm1(preds)
sample["count"] = np.clip(preds, 0, None)

# Save submission
sample.to_csv("./working/submission.csv", index=False)
