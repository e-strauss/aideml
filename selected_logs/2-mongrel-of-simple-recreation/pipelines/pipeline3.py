import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Load data
train = pd.read_csv("./input/train.csv", parse_dates=["datetime"])
test = pd.read_csv("./input/test.csv", parse_dates=["datetime"])
sample = pd.read_csv("./input/sampleSubmission.csv")


# Feature engineering
def create_features(df):
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    return df


train = create_features(train)
test = create_features(test)

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
    "day",
    "hour",
    "weekday",
]

X = train[features]
y = np.log1p(train["count"])
X_test = test[features]

# 5-Fold CV evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsles = []
for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_tr, y_tr)
    y_pred_val = np.clip(model.predict(X_val), 0, None)
    rmsle = mean_squared_error(y_val, y_pred_val, squared=False)
    rmsles.append(rmsle)

print(f"Mean RMSLE: {np.mean(rmsles):.5f}")

# Retrain on full data and predict test set
model_full = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    verbosity=0,
)
model_full.fit(X, y)
y_pred_test_log = np.clip(model_full.predict(X_test), 0, None)
y_pred_test = np.expm1(y_pred_test_log)

# Prepare submission
sample["count"] = np.round(y_pred_test).astype(int)
sample.to_csv("./working/submission.csv", index=False)
