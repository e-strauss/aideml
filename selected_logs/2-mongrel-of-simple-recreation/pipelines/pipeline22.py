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


# Feature engineering function
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

# Prepare log-target
y = np.log1p(train["count"].values)

# KFold target encoding for weather and season
kf = KFold(n_splits=5, shuffle=True, random_state=42)
weather_te = np.zeros(len(train))
season_te = np.zeros(len(train))
for tr_idx, val_idx in kf.split(train):
    # Weather encoding
    weather_map = pd.Series(y[tr_idx]).groupby(train.loc[tr_idx, "weather"]).mean()
    weather_te[val_idx] = train.loc[val_idx, "weather"].map(weather_map)
    # Season encoding
    season_map = pd.Series(y[tr_idx]).groupby(train.loc[tr_idx, "season"]).mean()
    season_te[val_idx] = train.loc[val_idx, "season"].map(season_map)

train["weather_te"] = weather_te
train["season_te"] = season_te

# Apply full-train encoding to test
global_weather_map = pd.Series(y).groupby(train["weather"]).mean()
global_season_map = pd.Series(y).groupby(train["season"]).mean()
test["weather_te"] = test["weather"].map(global_weather_map).fillna(y.mean())
test["season_te"] = test["season"].map(global_season_map).fillna(y.mean())

# Drop original categoricals
train = train.drop(["weather", "season", "count"], axis=1)
test = test.drop(["weather", "season"], axis=1)

# Feature list
features = [
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
    "weather_te",
    "season_te",
]

X = train[features].copy()
X_test = test[features].copy()

# 5-fold CV with LightGBM
rmsle_scores = []
for tr_idx, val_idx in KFold(n_splits=5, shuffle=True, random_state=42).split(X):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    model = lgb.LGBMRegressor(
        random_state=42,
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=40,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    model.fit(X_tr, y_tr)
    pred_log = model.predict(X_val)
    score = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(pred_log)))
    rmsle_scores.append(score)

cv_score = np.mean(rmsle_scores)
print(f"CV RMSLE: {cv_score:.5f}")

# Train final model and predict
final_model = lgb.LGBMRegressor(
    random_state=42,
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=40,
    subsample=0.8,
    colsample_bytree=0.8,
)
final_model.fit(X, y)
pred_log_test = final_model.predict(X_test)
pred_count = np.expm1(pred_log_test).clip(0)

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"datetime": test["datetime"], "count": pred_count})
submission.to_csv("./working/submission.csv", index=False)
