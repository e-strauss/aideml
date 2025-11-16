import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier


# AMEX metric implementation
def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    def top_four_percent_captured(y_true, y_pred):
        df = pd.concat([y_true, y_pred], axis=1).sort_values(
            "prediction", ascending=False
        )
        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        cutoff = int(0.04 * df["weight"].sum())
        df["wcum"] = df["weight"].cumsum()
        df_cut = df[df["wcum"] <= cutoff]
        return (df_cut["target"] == 1).sum() / (df["target"] == 1).sum()

    def weighted_gini(y_true, y_pred):
        df = pd.concat([y_true, y_pred], axis=1).sort_values(
            "prediction", ascending=False
        )
        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        df["random"] = (df["weight"] / df["weight"].sum()).cumsum()
        total_pos = (df["target"] * df["weight"]).sum()
        df["cum_pos"] = (df["target"] * df["weight"]).cumsum()
        df["lorentz"] = df["cum_pos"] / total_pos
        df["gini"] = (df["lorentz"] - df["random"]) * df["weight"]
        return df["gini"].sum()

    def normalized_weighted_gini(y_true, y_pred):
        df_true = y_true.rename(columns={"target": "prediction"})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, df_true)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)
    return 0.5 * (g + d)


# Load data
data = pd.read_csv("./input/train_data_downsampled.csv")
labels = pd.read_csv("./input/train_labels_downsampled.csv")

# Categorical cols and numeric features
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

# Compute EWM features
df_ewm = data.sort_values(["customer_ID", "S_2"]).copy()
df_ewm[features] = df_ewm.groupby("customer_ID")[features].transform(
    lambda x: x.ewm(alpha=0.3).mean()
)
df_last_ewm = df_ewm.groupby("customer_ID").tail(1).reset_index(drop=True)

# Compute last raw values
df_raw = data.sort_values(["customer_ID", "S_2"])
custs = df_last_ewm["customer_ID"].values
df_last_raw = (
    df_raw.groupby("customer_ID")
    .tail(1)
    .set_index("customer_ID")
    .loc[custs]
    .reset_index()
)

# Compute second‐last raw values (may be missing => NaN)
df_second = df_raw.groupby("customer_ID").nth(-2).reset_index().set_index("customer_ID")
df_second2 = df_second.reindex(index=custs).reset_index()

# Build feature matrices
X_ewm = df_last_ewm[features].reset_index(drop=True)
X_raw = df_last_raw[features].reset_index(drop=True).add_suffix("_raw")
# Delta = last_raw - second_last_raw, fill NaN with 0
X_delta = (
    df_last_raw[features].reset_index(drop=True)
    - df_second2[features].reset_index(drop=True)
).fillna(0)
X_delta = X_delta.add_suffix("_delta")

# Concatenate all features
X = pd.concat([X_ewm, X_raw, X_delta], axis=1)

# Align labels
y_df = labels.set_index("customer_ID").loc[custs].reset_index()
y = y_df["target"].values

# 5‐fold Stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_idx, val_idx in skf.split(X, y):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.1,
        depth=6,
        class_weights=[20, 1],
        verbose=0,
        random_seed=42,
    )
    model.fit(X_tr, y_tr)
    preds = model.predict_proba(X_val)[:, 1]
    score = amex_metric(
        pd.DataFrame({"target": y_val}), pd.DataFrame({"prediction": preds})
    )
    scores.append(score)

print(f"Mean AMEX metric with delta features: {np.mean(scores):.6f}")
