import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import os

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

# Compute global origin date for elapsed_days
dt_train = pd.to_datetime(train["datetime"])
dt_test = pd.to_datetime(test["datetime"])
origin_date = min(dt_train.min(), dt_test.min())


# Feature engineering
def add_datetime_features(df):
    dt = pd.to_datetime(df["datetime"])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.weekday
    # Cyclical features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    # Rush hour indicators
    df["is_morning_rush"] = df["hour"].isin([7, 8, 9]).astype(int)
    df["is_evening_rush"] = df["hour"].isin([16, 17, 18]).astype(int)
    return df


# Apply feature engineering
train = add_datetime_features(train)
test = add_datetime_features(test)

# Add elapsed_days feature
train["elapsed_days"] = (pd.to_datetime(train["datetime"]) - origin_date).dt.days
test["elapsed_days"] = (pd.to_datetime(test["datetime"]) - origin_date).dt.days

# One-hot encode weather and season
dummies_train = pd.get_dummies(
    train[["weather", "season"]].astype(str), prefix=["weather", "season"]
)
dummies_test = pd.get_dummies(
    test[["weather", "season"]].astype(str), prefix=["weather", "season"]
)
dummies_test = dummies_test.reindex(columns=dummies_train.columns, fill_value=0)

train = pd.concat([train, dummies_train], axis=1).drop(["weather", "season"], axis=1)
test = pd.concat([test, dummies_test], axis=1).drop(["weather", "season"], axis=1)

# Define features including elapsed_days
base_features = [
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "holiday",
    "workingday",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "weekday_sin",
    "weekday_cos",
    "is_morning_rush",
    "is_evening_rush",
    "elapsed_days",
]
features = base_features + list(dummies_train.columns)

X = train[features]
y = np.log1p(train["count"])
X_test = test[features]

# 5-fold Cross-Validation
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

# Train final model and predict on test
final_model = lgb.LGBMRegressor(random_state=42)
final_model.fit(X, y)
pred_log_test = final_model.predict(X_test)
pred_count = np.expm1(pred_log_test).clip(0)

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"datetime": test["datetime"], "count": pred_count})
submission.to_csv("./working/submission.csv", index=False)
