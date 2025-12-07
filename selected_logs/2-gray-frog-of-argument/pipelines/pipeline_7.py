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

# Drop unused columns but keep Postcode for target encoding
df = df.drop(
    columns=[
        "Transaction unique identifier",
        "Record Status - monthly file only",
        "Date of Transfer",
    ]
)

# Identify target and features (we'll handle Postcode separately)
target = "Price"
all_cols = [c for c in df.columns if c != target]
# Identify categorical columns (object dtype) except Postcode
cat_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != "Postcode"]

# Label-encode other categorical features
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Prepare for CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmses = []

# Static feature list excluding Postcode, target-encoded later
static_features = [c for c in all_cols if c != "Postcode"]

for fold, (train_idx, val_idx) in enumerate(kf.split(df), start=1):
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    # Compute postcode-level mean price on train fold
    postcode_means = train_df.groupby("Postcode")[target].mean()
    global_mean = train_df[target].mean()

    # Apply target encoding
    train_df["postcode_te"] = (
        train_df["Postcode"].map(postcode_means).fillna(global_mean)
    )
    val_df["postcode_te"] = val_df["Postcode"].map(postcode_means).fillna(global_mean)

    # Prepare training and validation data
    X_train = train_df[static_features + ["postcode_te"]].values
    y_train = train_df[target].values
    X_val = val_df[static_features + ["postcode_te"]].values
    y_val = val_df[target].values

    # Train model
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    rmses.append(rmse)
    print(f"Fold {fold} RMSE: {rmse:.2f}")

print(f"Average CV RMSE: {np.mean(rmses):.2f}")
