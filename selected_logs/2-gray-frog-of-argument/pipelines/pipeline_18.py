import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

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

# Label-encode all categorical features
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Base feature list (exclude target)
base_features = [c for c in df.columns if c != target]

# Prepare data for CV
X_df = df[base_features]
y = df[target].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_df), 1):
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    y_train = train_df[target].values
    y_val = val_df[target].values

    global_mean = y_train.mean()

    # K-Fold target encoding for 'District' and 'County'
    for col in ["District", "County"]:
        means = train_df.groupby(col)[target].mean()
        train_df[f"{col.lower()}_te"] = train_df[col].map(means).fillna(global_mean)
        val_df[f"{col.lower()}_te"] = val_df[col].map(means).fillna(global_mean)

    # Frequency encoding for 'District' and 'County'
    for col in ["District", "County"]:
        counts = train_df[col].value_counts()
        train_df[f"{col.lower()}_count"] = train_df[col].map(counts).fillna(0)
        val_df[f"{col.lower()}_count"] = val_df[col].map(counts).fillna(0)

    # Final feature list
    new_feats = [
        "district_te",
        "county_te",
        "district_count",
        "county_count",
    ]
    features = base_features + new_feats

    X_train = train_df[features].values
    X_val = val_df[features].values

    # Train LightGBM
    model = lgb.LGBMRegressor(
        learning_rate=0.05, n_estimators=1000, num_leaves=31, random_state=42
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        early_stopping_rounds=50,
        verbose=0,
    )
    preds = model.predict(X_val, num_iteration=model.best_iteration_)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    rmses.append(rmse)
    print(f"Fold {fold} RMSE: {rmse:.2f}")

print(f"Average CV RMSE: {np.mean(rmses):.2f}")
