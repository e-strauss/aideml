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

for fold, (train_idx, val_idx) in enumerate(kf.split(X_df), 1):
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    y_train = train_df[target].values
    y_val = val_df[target].values
    global_mean = y_train.mean()

    # Create year_month group
    train_df["year_month"] = train_df["year"] * 100 + train_df["month"]
    val_df["year_month"] = val_df["year"] * 100 + val_df["month"]

    # K-Fold target encoding for District, County, Town
    for col in ["District", "County", "Town"]:
        means = train_df.groupby(col)[target].mean()
        train_df[f"{col.lower()}_te"] = train_df[col].map(means).fillna(global_mean)
        val_df[f"{col.lower()}_te"] = val_df[col].map(means).fillna(global_mean)

    # Frequency encoding for Town, District, County
    for col in ["Town", "District", "County"]:
        counts = train_df[col].value_counts()
        train_df[f"{col.lower()}_count"] = train_df[col].map(counts).fillna(0)
        val_df[f"{col.lower()}_count"] = val_df[col].map(counts).fillna(0)

    # K-Fold target encoding for year_month
    ym_means = train_df.groupby("year_month")[target].mean()
    train_df["year_month_te"] = train_df["year_month"].map(ym_means).fillna(global_mean)
    val_df["year_month_te"] = val_df["year_month"].map(ym_means).fillna(global_mean)

    # Final feature list
    features = base_features + [
        "district_te",
        "county_te",
        "town_te",
        "town_count",
        "district_count",
        "county_count",
        "year_month_te",
    ]

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
