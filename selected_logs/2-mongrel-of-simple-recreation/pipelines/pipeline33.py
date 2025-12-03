import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import os

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

# Origin date for elapsed_days
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
    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    # Rush hour
    df["is_morning_rush"] = df["hour"].isin([7, 8, 9]).astype(int)
    df["is_evening_rush"] = df["hour"].isin([16, 17, 18]).astype(int)
    # Elapsed days
    df["elapsed_days"] = (dt - origin_date).dt.days
    return df


train = add_datetime_features(train)
test = add_datetime_features(test)

# One-hot weather and season
d_tr = pd.get_dummies(
    train[["weather", "season"]].astype(str), prefix=["weather", "season"]
)
d_te = pd.get_dummies(
    test[["weather", "season"]].astype(str), prefix=["weather", "season"]
)
d_te = d_te.reindex(columns=d_tr.columns, fill_value=0)
train = pd.concat([train, d_tr], axis=1).drop(["weather", "season"], axis=1)
test = pd.concat([test, d_te], axis=1).drop(["weather", "season"], axis=1)

# Prepare data
features = [
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
] + list(d_tr.columns)

X = train[features]
y = np.log1p(train["count"])
X_test = test[features]

# CV with early stopping and stronger regularization
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []
best_iters = []
for tr_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    model = lgb.LGBMRegressor(
        random_state=42,
        learning_rate=0.03,
        num_leaves=60,
        n_estimators=2000,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        early_stopping_rounds=50,
        verbose=False,
    )
    best_iter = model.best_iteration_
    best_iters.append(best_iter)
    pred_log = model.predict(X_val, num_iteration=best_iter)
    score = np.sqrt(mean_squared_error(y_val, pred_log))
    rmsle_scores.append(score)

cv_score = np.mean(rmsle_scores)
print(f"CV RMSLE: {cv_score:.5f}")

# Final model using average best iterations
final_n = int(np.mean(best_iters))
final_model = lgb.LGBMRegressor(
    random_state=42,
    learning_rate=0.03,
    num_leaves=60,
    n_estimators=final_n,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
)
final_model.fit(X, y)
pred_log_test = final_model.predict(X_test)
pred_count = np.expm1(pred_log_test).clip(0)

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"datetime": test["datetime"], "count": pred_count})
submission.to_csv("./working/submission.csv", index=False)
