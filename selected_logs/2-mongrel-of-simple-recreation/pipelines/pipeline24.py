import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb

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
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["is_morning_rush"] = df["hour"].isin([7, 8, 9]).astype(int)
    df["is_evening_rush"] = df["hour"].isin([16, 17, 18]).astype(int)
    return df


train = add_datetime_features(train)
test = add_datetime_features(test)

train["elapsed_days"] = (pd.to_datetime(train["datetime"]) - origin_date).dt.days
test["elapsed_days"] = (pd.to_datetime(test["datetime"]) - origin_date).dt.days

# One-hot encode weather and season
dtr = pd.get_dummies(
    train[["weather", "season"]].astype(str), prefix=["weather", "season"]
)
dte = pd.get_dummies(
    test[["weather", "season"]].astype(str), prefix=["weather", "season"]
)
dte = dte.reindex(columns=dtr.columns, fill_value=0)

train = pd.concat([train, dtr], axis=1).drop(["weather", "season"], axis=1)
test = pd.concat([test, dte], axis=1).drop(["weather", "season"], axis=1)

# Define features
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
features = base_features + list(dtr.columns)

X = train[features]
y = train["count"]
X_test = test[features]

# LightGBM parameters
params = {
    "objective": "poisson",
    "metric": "rmse",
    "learning_rate": 0.05,
    "verbose": -1,
    "seed": 42,
}

# 5-fold CV with early stopping via lgb.train
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []
best_iters = []

for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=10000,
        valid_sets=[dval],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    best_iter = booster.best_iteration
    best_iters.append(best_iter)
    pred_val = booster.predict(X_val, num_iteration=best_iter).clip(0)
    score = np.sqrt(mean_squared_log_error(y_val, pred_val))
    rmsle_scores.append(score)

cv_score = np.mean(rmsle_scores)
print(f"CV RMSLE: {cv_score:.5f}")

# Retrain final model using average best iteration
avg_iter = int(np.mean(best_iters))
dall = lgb.Dataset(X, label=y)
final_booster = lgb.train(params, dall, num_boost_round=avg_iter, verbose_eval=False)

pred_test = final_booster.predict(X_test, num_iteration=avg_iter).clip(0)

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"datetime": test["datetime"], "count": pred_test})
submission.to_csv("./working/submission.csv", index=False)
