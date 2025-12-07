import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Load data
df = pd.read_csv(
    "./input/price_paid_records.csv", parse_dates=["Date of Transfer"], low_memory=False
)

# Feature engineering
df["year"] = df["Date of Transfer"].dt.year
df["month"] = df["Date of Transfer"].dt.month
df["day"] = df["Date of Transfer"].dt.day
df = df.drop(
    [
        "Date of Transfer",
        "Transaction unique identifier",
        "Record Status - monthly file only",
    ],
    axis=1,
)

# Separate target
y = df["Price"].values
X = df.drop("Price", axis=1)

# Label encode categorical features
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

X = X.values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmses = []

for train_idx, valid_idx in kf.split(X):
    X_train, X_valid = X[train_idx], X[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    model = XGBRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, preds, squared=False)
    rmses.append(rmse)
    print(f"Fold RMSE: {rmse:.4f}")

print(f"Average CV RMSE: {np.mean(rmses):.4f}")
