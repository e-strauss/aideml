import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import os

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


train = add_datetime_features(train)
test = add_datetime_features(test)

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
]
features = base_features + list(dummies_train.columns)

# Prepare for CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []
df = train.copy().reset_index(drop=True)

for train_idx, val_idx in kf.split(df):
    # Split
    tr = df.loc[train_idx].copy()
    va = df.loc[val_idx].copy()
    # Compute aggregates on training fold
    hour_mean = tr.groupby("hour")["count"].mean()
    wd_mean = tr.groupby("weekday")["count"].mean()
    global_mean = tr["count"].mean()
    # Map to features
    for sub in (tr, va):
        sub["hour_avg_count"] = sub["hour"].map(hour_mean).fillna(global_mean)
        sub["weekday_avg_count"] = sub["weekday"].map(wd_mean).fillna(global_mean)
    # Prepare X/y
    X_tr = tr[features + ["hour_avg_count", "weekday_avg_count"]]
    X_val = va[features + ["hour_avg_count", "weekday_avg_count"]]
    y_tr = np.log1p(tr["count"])
    y_val = np.log1p(va["count"])
    # Train
    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X_tr, y_tr)
    pred_log = model.predict(X_val)
    # Score
    score = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(pred_log)))
    rmsle_scores.append(score)

cv_score = np.mean(rmsle_scores)
print(f"CV RMSLE: {cv_score:.5f}")

# Full model for submission
# Compute aggregate on full train
hour_mean_full = df.groupby("hour")["count"].mean()
wd_mean_full = df.groupby("weekday")["count"].mean()
global_mean_full = df["count"].mean()
# Add to train & test
train["hour_avg_count"] = train["hour"].map(hour_mean_full).fillna(global_mean_full)
train["weekday_avg_count"] = train["weekday"].map(wd_mean_full).fillna(global_mean_full)
test["hour_avg_count"] = test["hour"].map(hour_mean_full).fillna(global_mean_full)
test["weekday_avg_count"] = test["weekday"].map(wd_mean_full).fillna(global_mean_full)
# Final train/predict
X_full = train[features + ["hour_avg_count", "weekday_avg_count"]]
y_full = np.log1p(train["count"])
X_test = test[features + ["hour_avg_count", "weekday_avg_count"]]
final_model = lgb.LGBMRegressor(random_state=42)
final_model.fit(X_full, y_full)
pred_log_test = final_model.predict(X_test)
pred_count = np.expm1(pred_log_test).clip(0)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"datetime": test["datetime"], "count": pred_count})
submission.to_csv("./working/submission.csv", index=False)
