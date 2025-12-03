import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
sample = pd.read_csv("./input/sampleSubmission.csv")

# Feature engineering: parse datetime
for df in [train, test]:
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday

# Mark train/test and combine for consistent one-hot encoding
train["is_train"] = 1
test["is_train"] = 0
combined = pd.concat([train, test], sort=False)

# One-hot encode season and weather
combined = pd.get_dummies(combined, columns=["season", "weather"], drop_first=False)

# Split back to processed train and test
train_proc = combined[combined["is_train"] == 1].drop(
    ["is_train", "datetime", "casual", "registered"], axis=1
)
test_proc = combined[combined["is_train"] == 0].drop(
    ["is_train", "datetime", "casual", "registered", "count"], axis=1
)

# Prepare features and target
X = train_proc.drop("count", axis=1)
y = train_proc["count"]
y_log = np.log1p(y)

# 5-fold cross-validation with RandomForest
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsles = []
for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    pred_val_log = model.predict(X_val)
    rmsle = np.sqrt(mean_squared_error(y_val, pred_val_log))
    rmsles.append(rmsle)

print(f"Mean RMSLE: {np.mean(rmsles):.5f}")

# Retrain on full data and predict on test
final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
final_model.fit(X, y_log)
pred_test_log = final_model.predict(test_proc)
pred_test = np.expm1(pred_test_log)
pred_test = np.clip(pred_test, a_min=0, a_max=None)

# Save submission
sample["count"] = pred_test
sample.to_csv("./working/submission.csv", index=False)
