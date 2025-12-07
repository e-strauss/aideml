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

# Extract outward postcode sector for finer-grained location feature
df["postcode_out"] = df["Postcode"].astype(str).str.split().str[0]

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
# Label-encode categorical features
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

    # K-Fold target encoding for 'District'
    district_means = train_df.groupby("District")[target].mean()
    train_df["district_te"] = (
        train_df["District"].map(district_means).fillna(global_mean)
    )
    val_df["district_te"] = val_df["District"].map(district_means).fillna(global_mean)

    # K-Fold target encoding for 'County'
    county_means = train_df.groupby("County")[target].mean()
    train_df["county_te"] = train_df["County"].map(county_means).fillna(global_mean)
    val_df["county_te"] = val_df["County"].map(county_means).fillna(global_mean)

    # K-Fold target encoding for 'postcode_out'
    postcode_means = train_df.groupby("postcode_out")[target].mean()
    train_df["postcode_te"] = (
        train_df["postcode_out"].map(postcode_means).fillna(global_mean)
    )
    val_df["postcode_te"] = (
        val_df["postcode_out"].map(postcode_means).fillna(global_mean)
    )

    # Frequency encoding for 'District'
    district_counts = train_df["District"].value_counts()
    train_df["district_count"] = train_df["District"].map(district_counts).fillna(0)
    val_df["district_count"] = val_df["District"].map(district_counts).fillna(0)

    # Frequency encoding for 'County'
    county_counts = train_df["County"].value_counts()
    train_df["county_count"] = train_df["County"].map(county_counts).fillna(0)
    val_df["county_count"] = val_df["County"].map(county_counts).fillna(0)

    # Frequency encoding for 'postcode_out'
    postcode_counts = train_df["postcode_out"].value_counts()
    train_df["postcode_count"] = train_df["postcode_out"].map(postcode_counts).fillna(0)
    val_df["postcode_count"] = val_df["postcode_out"].map(postcode_counts).fillna(0)

    # Final feature list
    features = base_features + [
        "district_te",
        "county_te",
        "postcode_te",
        "district_count",
        "county_count",
        "postcode_count",
    ]
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
