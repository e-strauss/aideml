import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load only needed columns and parse date
usecols = [
    "Date of Transfer",
    "County",
    "District",
    "Duration",
    "Old/New",
    "PPDCategory Type",
    "Property Type",
    "Town/City",
    "Price",
]
df = pd.read_csv(
    "input/price_paid_records.csv",
    usecols=usecols,
    parse_dates=["Date of Transfer"],
    low_memory=False,
)

# Feature engineering
df["year"] = df["Date of Transfer"].dt.year
df["month"] = df["Date of Transfer"].dt.month
df.drop(["Date of Transfer"], axis=1, inplace=True)

# Convert to categorical
cat_cols = [
    "County",
    "District",
    "Duration",
    "Old/New",
    "PPDCategory Type",
    "Property Type",
    "Town/City",
]
for col in cat_cols:
    df[col] = df[col].astype("category")

# Prepare data
X = df.drop("Price", axis=1)
y = df["Price"]

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmses = []

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model = lgb.LGBMRegressor(objective="regression", n_estimators=100, n_jobs=-1)
    model.fit(
        X_train,
        y_train,
        categorical_feature=cat_cols,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=False,
    )
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    rmses.append(rmse)

print(f"CV RMSE: {np.mean(rmses):.4f}")
