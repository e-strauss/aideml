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
orig_cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in orig_cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Base features (all except target)
base_features = [c for c in df.columns if c != target]

# Prepare data
X_df = df[base_features]
y = df[target].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmses = []
alpha = 20

for fold, (train_idx, val_idx) in enumerate(kf.split(X_df), 1):
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    y_train = train_df[target].values
    y_val = val_df[target].values
    global_mean = y_train.mean()

    # Smoothed target encoding for District, County, Town
    for feat in ["District", "County", "Town"]:
        stats = train_df.groupby(feat)[target].agg(["mean", "count"])
        counts = stats["count"]
        means = stats["mean"]
        smooth = (means * counts + global_mean * alpha) / (counts + alpha)
        train_df[f"{feat.lower()}_te"] = train_df[feat].map(smooth).fillna(global_mean)
        val_df[f"{feat.lower()}_te"] = val_df[feat].map(smooth).fillna(global_mean)

    # Frequency encoding for Town, District, County
    for feat in ["Town", "District", "County"]:
        freq = train_df[feat].value_counts()
        train_df[f"{feat.lower()}_count"] = train_df[feat].map(freq).fillna(0)
        val_df[f"{feat.lower()}_count"] = val_df[feat].map(freq).fillna(0)

    # Final feature list
    features = base_features + [
        "district_te",
        "county_te",
        "town_te",
        "town_count",
        "district_count",
        "county_count",
    ]
    X_train = train_df[features].values
    X_val = val_df[features].values

    # LightGBM training
    model = lgb.LGBMRegressor(
        learning_rate=0.05,
        n_estimators=1000,
        num_leaves=31,
        random_state=42,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
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
