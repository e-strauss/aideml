import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load a subset for speed
df = pd.read_csv("./input/price_paid_records.csv", nrows=500000)

# Parse dates
df["Date of Transfer"] = pd.to_datetime(df["Date of Transfer"])
df["year"] = df["Date of Transfer"].dt.year
df["month"] = df["Date of Transfer"].dt.month
df["day"] = df["Date of Transfer"].dt.day

# Drop unused columns
df = df.drop(
    [
        "Transaction unique identifier",
        "Record Status - monthly file only",
        "Date of Transfer",
    ],
    axis=1,
)

# Identify target and features
y = df["Price"].values
X = df.drop("Price", axis=1)

# Label‐encode categoricals
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

X = X.values

# 5‐fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmses = []
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    rmses.append(rmse)

print(f"Average CV RMSE: {np.mean(rmses):.2f}")
