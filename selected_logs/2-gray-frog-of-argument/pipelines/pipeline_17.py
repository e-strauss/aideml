import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load a sample of the data for a fast baseline
DATA_PATH = "./input/price_paid_records.csv"
NROWS = 2_000_000  # sample first 2M rows for speed
df = pd.read_csv(DATA_PATH, parse_dates=["Date of Transfer"], nrows=NROWS)

# Feature engineering: extract year and month
df["year"] = df["Date of Transfer"].dt.year
df["month"] = df["Date of Transfer"].dt.month

# Drop unused columns
df = df.drop(
    columns=[
        "Transaction unique identifier",
        "Record Status - monthly file only",
        "Date of Transfer",
    ]
)

# Identify target and features
target = "Price"
all_features = [c for c in df.columns if c != target]

# Label-encode all categorical features
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Prepare data arrays
X = df[all_features].values
y = df[target].values

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Train model
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    rmses.append(rmse)
    print(f"Fold {fold} RMSE: {rmse:.2f}")

print(f"Average CV RMSE: {np.mean(rmses):.2f}")
