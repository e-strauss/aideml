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
df = df.drop(
    columns=[
        "Transaction unique identifier",
        "Record Status - monthly file only",
        "Date of Transfer",
    ]
)

# Prepare features and target
target = "Price"
features = [c for c in df.columns if c != target]

# Label-encode categorical features
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df[features].values
y_orig = df[target].values
y = np.log1p(y_orig)  # log1p transform target

# 5-fold cross-validation with HistGradientBoostingRegressor on log-target
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val_log = y[train_idx], y[val_idx]
    y_val_orig = y_orig[val_idx]

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds_log = model.predict(X_val)
    preds = np.expm1(preds_log)  # back-transform

    rmse = np.sqrt(mean_squared_error(y_val_orig, preds))
    rmses.append(rmse)
    print(f"Fold {fold} RMSE: {rmse:.2f}")

print(f"Average CV RMSE: {np.mean(rmses):.2f}")
