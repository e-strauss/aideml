import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Load a sample of the data for speed
DATA_PATH = "./input/price_paid_records.csv"
NROWS = 2_000_000
df = pd.read_csv(DATA_PATH, parse_dates=["Date of Transfer"], nrows=NROWS)

# Fix column name: rename Town/City to Town
df.rename(columns={"Town/City": "Town"}, inplace=True)

# Basic date features
df["year"] = df["Date of Transfer"].dt.year
df["month"] = df["Date of Transfer"].dt.month

# Drop unused columns
df.drop(
    columns=[
        "Transaction unique identifier",
        "Record Status - monthly file only",
        "Date of Transfer",
    ],
    inplace=True,
)

# Identify target and encode categoricals
target = "Price"
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Base features (all except target)
base_features = [c for c in df.columns if c != target]

# Prepare data
X_df = df[base_features]
y = df[target].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmses = []
alpha = 100.0  # smoothing parameter for postcode TE

for fold, (train_idx, val_idx) in enumerate(kf.split(X_df), 1):
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    y_train = train_df[target].values
    y_val = val_df[target].values
    global_mean = y_train.mean()

    # K-Fold target encoding for District
    district_means = train_df.groupby("District")[target].mean()
    train_df["district_te"] = (
        train_df["District"].map(district_means).fillna(global_mean)
    )
    val_df["district_te"] = val_df["District"].map(district_means).fillna(global_mean)

    # K-Fold target encoding for County
    county_means = train_df.groupby("County")[target].mean()
    train_df["county_te"] = train_df["County"].map(county_means).fillna(global_mean)
    val_df["county_te"] = val_df["County"].map(county_means).fillna(global_mean)

    # Frequency encoding for Town
    town_counts = train_df["Town"].value_counts()
    train_df["town_count"] = train_df["Town"].map(town_counts).fillna(0)
    val_df["town_count"] = val_df["Town"].map(town_counts).fillna(0)

    # Smoothed target encoding for Postcode
    post_stats = train_df.groupby("Postcode")[target].agg(["mean", "count"])
    mean_map = post_stats["mean"]
    count_map = post_stats["count"]
    # Compute smoothed encoding
    train_counts = train_df["Postcode"].map(count_map).fillna(0)
    train_means = train_df["Postcode"].map(mean_map).fillna(global_mean)
    train_df["postcode_te"] = (train_means * train_counts + alpha * global_mean) / (
        train_counts + alpha
    )
    train_df["postcode_te"].fillna(global_mean, inplace=True)
    val_counts = val_df["Postcode"].map(count_map).fillna(0)
    val_means = val_df["Postcode"].map(mean_map).fillna(global_mean)
    val_df["postcode_te"] = (val_means * val_counts + alpha * global_mean) / (
        val_counts + alpha
    )
    val_df["postcode_te"].fillna(global_mean, inplace=True)

    # Final feature list
    features = base_features + ["district_te", "county_te", "town_count", "postcode_te"]
    X_train = train_df[features].values
    X_val = val_df[features].values

    # LightGBM training
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
