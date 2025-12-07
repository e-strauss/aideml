import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Load data
DATA_PATH = "./input/price_paid_records.csv"
NROWS = 2_000_000
df = pd.read_csv(DATA_PATH, parse_dates=["Date of Transfer"], nrows=NROWS)

# Rename column
df.rename(columns={"Town/City": "Town"}, inplace=True)

# Date features
df["year"] = df["Date of Transfer"].dt.year
df["month"] = df["Date of Transfer"].dt.month

# Drop unused
df.drop(
    columns=[
        "Transaction unique identifier",
        "Record Status - monthly file only",
        "Date of Transfer",
    ],
    inplace=True,
)

# Label‐encode categoricals
target = "Price"
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Base features
base_features = [c for c in df.columns if c != target]

# CV setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmses = []
alpha = 20  # smoothing parameter

for fold, (train_idx, val_idx) in enumerate(kf.split(df), 1):
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    y_train = train_df[target].values
    y_val = val_df[target].values
    global_mean = y_train.mean()

    # Smoothed target encoding for District and County
    for col in ["District", "County"]:
        agg = train_df.groupby(col)[target].agg(["mean", "count"])
        smoothing = (agg["mean"] * agg["count"] + global_mean * alpha) / (
            agg["count"] + alpha
        )
        mapping = smoothing.to_dict()
        train_df[f"{col}_smooth_te"] = train_df[col].map(mapping).fillna(global_mean)
        val_df[f"{col}_smooth_te"] = val_df[col].map(mapping).fillna(global_mean)

    # Frequency encoding for Town
    town_counts = train_df["Town"].value_counts()
    train_df["town_count"] = train_df["Town"].map(town_counts).fillna(0)
    val_df["town_count"] = val_df["Town"].map(town_counts).fillna(0)

    # Features and data
    features = base_features + ["District_smooth_te", "County_smooth_te", "town_count"]
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
