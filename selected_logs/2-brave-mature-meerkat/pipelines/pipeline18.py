import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier


# AMEX metric implementation
def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    def top_four_percent_captured(df_true, df_pred):
        df = pd.concat([df_true, df_pred], axis=1).sort_values(
            "prediction", ascending=False
        )
        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        cutoff = int(0.04 * df["weight"].sum())
        df["wcum"] = df["weight"].cumsum()
        df_cut = df[df["wcum"] <= cutoff]
        return (df_cut["target"] == 1).sum() / (df["target"] == 1).sum()

    def weighted_gini(df_true, df_pred):
        df = pd.concat([df_true, df_pred], axis=1).sort_values(
            "prediction", ascending=False
        )
        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        df["random"] = (df["weight"] / df["weight"].sum()).cumsum()
        total_pos = (df["target"] * df["weight"]).sum()
        df["cum_pos"] = (df["target"] * df["weight"]).cumsum()
        df["lorentz"] = df["cum_pos"] / total_pos
        df["gini"] = (df["lorentz"] - df["random"]) * df["weight"]
        return df["gini"].sum()

    def normalized_weighted_gini(df_true, df_pred):
        df_true_perm = df_true.rename(columns={"target": "prediction"})
        return weighted_gini(df_true, df_pred) / weighted_gini(df_true, df_true_perm)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)
    return 0.5 * (g + d)


# Load data
data = pd.read_csv("./input/train_data_downsampled.csv")
labels = pd.read_csv("./input/train_labels_downsampled.csv")

# Define categorical columns and numeric features
cat_cols = [
    "S_2",
    "B_30",
    "B_38",
    "D_114",
    "D_116",
    "D_117",
    "D_120",
    "D_126",
    "D_63",
    "D_64",
    "D_66",
    "D_68",
]
features = [c for c in data.columns if c not in ["customer_ID"] + cat_cols]

# Exponentially weighted means per customer
df_ewm = data.sort_values(["customer_ID", "S_2"]).copy()
df_ewm[features] = df_ewm.groupby("customer_ID")[features].transform(
    lambda x: x.ewm(alpha=0.3).mean()
)
df_last_ewm = df_ewm.groupby("customer_ID").tail(1).reset_index(drop=True)

# Last raw values per customer
df_raw = data.sort_values(["customer_ID", "S_2"]).copy()
df_last_raw = df_raw.groupby("customer_ID").tail(1).reset_index(drop=True)

# Std dev features per customer
df_std = data.groupby("customer_ID")[features].std().reset_index()
df_last_std = pd.merge(
    df_last_ewm[["customer_ID"]], df_std, on="customer_ID", how="left"
)
X_std = df_last_std[features].fillna(0).add_suffix("_std").reset_index(drop=True)

# Frequency encoding for categoricals from last raw
freq_maps = {col: data[col].value_counts(normalize=True).to_dict() for col in cat_cols}
X_cat_freq = pd.DataFrame(
    {
        f"{col}_freq": df_last_raw[col].map(freq_maps[col]).fillna(0).values
        for col in cat_cols
    }
)

# Median features per customer
df_median = data.groupby("customer_ID")[features].median().reset_index()
df_median_aligned = (
    df_median.set_index("customer_ID")
    .loc[df_last_ewm["customer_ID"]]
    .reset_index(drop=True)
)
X_median = df_median_aligned[features].fillna(0).add_suffix("_median")

# Base EWM and raw features
X_ewm = df_last_ewm[features].reset_index(drop=True)
X_raw = df_last_raw[features].reset_index(drop=True).add_suffix("_raw")

# Residual features
X_resid = pd.DataFrame(
    {f"{f}_resid": X_raw[f"{f}_raw"].values - X_ewm[f].values for f in features}
)

# Min and Max features per customer
df_min = data.groupby("customer_ID")[features].min().reset_index()
df_max = data.groupby("customer_ID")[features].max().reset_index()
df_min_max = pd.merge(df_min, df_max, on="customer_ID", suffixes=("_min", "_max"))
df_min_max = (
    df_min_max.set_index("customer_ID")
    .loc[df_last_ewm["customer_ID"]]
    .reset_index(drop=True)
)
X_min = df_min_max[[f"{f}_min" for f in features]]
X_max = df_min_max[[f"{f}_max" for f in features]]

# Delta features: difference between last and second-last raw values
df_prev = df_raw[features].groupby(df_raw["customer_ID"]).shift(1)
df_temp = df_raw.copy()
for f in features:
    df_temp[f + "_prev"] = df_prev[f]
df_last2 = df_temp.groupby("customer_ID").tail(1).reset_index(drop=True)
delta_data = {f"{f}_delta": df_last2[f] - df_last2[f + "_prev"] for f in features}
X_delta = pd.DataFrame(delta_data).fillna(0)

# Combine all features including new median
X = pd.concat(
    [X_ewm, X_raw, X_std, X_cat_freq, X_median, X_resid, X_min, X_max, X_delta], axis=1
)

# Align labels
y_df = (
    labels.set_index("customer_ID")
    .loc[df_last_ewm["customer_ID"]]
    .reset_index(drop=True)
)
y = y_df["target"].values

# 5-fold Stratified CV with CatBoost
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        class_weights=[20, 1],
        verbose=0,
        random_seed=42,
        early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    preds = model.predict_proba(X_val)[:, 1]
    score = amex_metric(
        pd.DataFrame({"target": y_val}), pd.DataFrame({"prediction": preds})
    )
    scores.append(score)

print(f"Mean AMEX metric: {np.mean(scores):.6f}")
