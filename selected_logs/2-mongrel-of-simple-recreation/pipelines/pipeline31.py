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
    df["dayofyear"] = dt.dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    df["is_morning_rush"] = df["hour"].isin([7, 8, 9]).astype(int)
    df["is_evening_rush"] = df["hour"].isin([16, 17, 18]).astype(int)
    return df


train = add_datetime_features(train)
test = add_datetime_features(test)

# Add elapsed_days
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

# Prepare features
base_features = [
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "dayofyear",
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
    "dayofyear_sin",
    "dayofyear_cos",
    "is_morning_rush",
    "is_evening_rush",
    "elapsed_days",
]
features = base_features + list(dummies_train.columns)
X = train[features]
X_test = test[features]

# Prepare target for Poisson (shift to avoid zeros)
y_count = train["count"]
y_poisson = y_count + 1.0

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []
for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr = y_poisson.iloc[train_idx]
    y_val_count = y_count.iloc[val_idx]
    model = lgb.LGBMRegressor(
        objective="poisson",
        random_state=42,
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=40,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    model.fit(X_tr, y_tr)
    pred_poisson = model.predict(X_val)
    pred_count = (pred_poisson - 1.0).clip(0)
    score = np.sqrt(mean_squared_log_error(y_val_count, pred_count))
    rmsle_scores.append(score)

cv_score = np.mean(rmsle_scores)
print(f"CV RMSLE: {cv_score:.5f}")

# Train final model on full data
final_model = lgb.LGBMRegressor(
    objective="poisson",
    random_state=42,
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=40,
    subsample=0.8,
    colsample_bytree=0.8,
)
final_model.fit(X, y_poisson)
pred_poisson_test = final_model.predict(X_test)
pred_count_test = (pred_poisson_test - 1.0).clip(0)

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"datetime": test["datetime"], "count": pred_count_test})
submission.to_csv("./working/submission.csv", index=False)
