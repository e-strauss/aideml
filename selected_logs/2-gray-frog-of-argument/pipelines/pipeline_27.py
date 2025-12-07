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

# Rename and extract date features
df.rename(columns={"Town/City": "Town"}, inplace=True)
df["year"] = df["Date of Transfer"].dt.year
df["month"] = df["Date of Transfer"].dt.month
df["weekday"] = df["Date of Transfer"].dt.weekday
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

# Drop unused columns
df.drop(
    columns=[
        "Transaction unique identifier",
        "Record Status - monthly file only",
        "Date of Transfer",
    ],
    inplace=True,
)

# Label encode categoricals
target = "Price"
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Prepare base features and target
base_features = [c for c in df.columns if c != target]
X_df = df[base_features]
y = df[target].values

# 5-fold CV with LightGBM
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_df), 1):
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    y_train = train_df[target].values
    y_val = val_df[target].values
    global_mean = y_train.mean()

    # Target and frequency encoding for District, County, Town
    for col in ["District", "County", "Town"]:
        means = train_df.groupby(col)[target].mean()
        train_df[f"{col.lower()}_te"] = train_df[col].map(means).fillna(global_mean)
        val_df[f"{col.lower()}_te"] = val_df[col].map(means).fillna(global_mean)
        counts = train_df[col].value_counts()
        train_df[f"{col.lower()}_count"] = train_df[col].map(counts).fillna(0)
        val_df[f"{col.lower()}_count"] = val_df[col].map(counts).fillna(0)

    features = (
        base_features
        + [f"{col.lower()}_te" for col in ["District", "County", "Town"]]
        + [f"{col.lower()}_count" for col in ["District", "County", "Town"]]
    )

    X_train = train_df[features].values
    X_val = val_df[features].values

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
