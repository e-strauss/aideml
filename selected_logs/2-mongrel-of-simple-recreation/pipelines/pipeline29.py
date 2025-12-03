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
    df["dayofyear"] = dt.dt.dayofyear
    # cyclical features
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

# elapsed_days
origin = min(
    pd.to_datetime(train["datetime"]).min(), pd.to_datetime(test["datetime"]).min()
)
train["elapsed_days"] = (pd.to_datetime(train["datetime"]) - origin).dt.days
test["elapsed_days"] = (pd.to_datetime(test["datetime"]) - origin).dt.days

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

# Prepare base features
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
] + list(dtr.columns)

X_base = train[base_features].copy()
X_test_base = test[base_features].copy()
y = np.log1p(train["count"].values)

# 5-fold CV with target encoding for hour and weekday
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []

for train_idx, val_idx in kf.split(X_base):
    X_tr_b, X_val_b = X_base.iloc[train_idx], X_base.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    # Compute mappings
    df_tr = pd.DataFrame(
        {"hour": X_tr_b["hour"], "weekday": X_tr_b["weekday"], "y": y_tr}
    )
    hm = df_tr.groupby("hour")["y"].mean().to_dict()
    wm = df_tr.groupby("weekday")["y"].mean().to_dict()
    dh, dw = np.mean(list(hm.values())), np.mean(list(wm.values()))
    # Apply TE
    X_tr = X_tr_b.copy()
    X_tr["hour_te"] = X_tr["hour"].map(hm)
    X_tr["weekday_te"] = X_tr["weekday"].map(wm)
    X_val = X_val_b.copy()
    X_val["hour_te"] = X_val["hour"].map(hm).fillna(dh)
    X_val["weekday_te"] = X_val["weekday"].map(wm).fillna(dw)
    # Train
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

# Final target encoding on full data
df_full = pd.DataFrame({"hour": X_base["hour"], "weekday": X_base["weekday"], "y": y})
hm_f = df_full.groupby("hour")["y"].mean().to_dict()
wm_f = df_full.groupby("weekday")["y"].mean().to_dict()
dh_f, dw_f = np.mean(list(hm_f.values())), np.mean(list(wm_f.values()))

X_full = X_base.copy()
X_full["hour_te"] = X_full["hour"].map(hm_f).fillna(dh_f)
X_full["weekday_te"] = X_full["weekday"].map(wm_f).fillna(dw_f)
X_test = X_test_base.copy()
X_test["hour_te"] = X_test["hour"].map(hm_f).fillna(dh_f)
X_test["weekday_te"] = X_test["weekday"].map(wm_f).fillna(dw_f)

# Train final model and predict
final_model = lgb.LGBMRegressor(
    random_state=42,
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=40,
    subsample=0.8,
    colsample_bytree=0.8,
)
final_model.fit(X_full, y)
pred_log_test = final_model.predict(X_test)
pred_count = np.expm1(pred_log_test).clip(0)

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"datetime": test["datetime"], "count": pred_count})
submission.to_csv("./working/submission.csv", index=False)
