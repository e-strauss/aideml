import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Load a subset of the data for speed
df = pd.read_csv("./input/price_paid_records.csv", nrows=2_000_000)

# Feature engineering: parse date
df["Date of Transfer"] = pd.to_datetime(df["Date of Transfer"])
df["year"] = df["Date of Transfer"].dt.year
df["month"] = df["Date of Transfer"].dt.month
df["day"] = df["Date of Transfer"].dt.day

# Drop unused or constant columns
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

X = X.values  # convert to numpy array

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmses = []

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = Ridge()
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    rmses.append(rmse)

print(f"CV RMSE: {np.mean(rmses):.2f}")
