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

# Compute EWM features
df_ewm = data.sort_values(["customer_ID", "S_2"]).copy()
df_ewm[features] = df_ewm.groupby("customer_ID")[features].transform(
    lambda x: x.ewm(alpha=0.3).mean()
)
df_last_ewm = df_ewm.groupby("customer_ID").tail(1).reset_index(drop=True)

# Compute last raw values
df_raw = data.sort_values(["customer_ID", "S_2"])
df_last_raw = df_raw.groupby("customer_ID").tail(1).reset_index(drop=True)

# Compute per-customer std features
df_std = data.groupby("customer_ID")[features].std().reset_index()
df_last_std = pd.merge(
    df_last_ewm[["customer_ID"]], df_std, on="customer_ID", how="left"
)
X_std = df_last_std[features].fillna(0).add_suffix("_std").reset_index(drop=True)

# Compute statement count per customer
df_count = data.groupby("customer_ID").size().reset_index(name="stmt_count")
df_cnt = pd.merge(df_last_ewm[["customer_ID"]], df_count, on="customer_ID", how="left")
X_cnt = df_cnt["stmt_count"].reset_index(drop=True)

# Build feature matrices
X_ewm = df_last_ewm[features].reset_index(drop=True)
X_raw = df_last_raw[features].reset_index(drop=True).add_suffix("_raw")
X = pd.concat([X_ewm, X_raw, X_std], axis=1)
X["stmt_count"] = X_cnt

# Align labels
y_df = (
    labels.set_index("customer_ID")
    .loc[df_last_ewm["customer_ID"]]
    .reset_index(drop=True)
)
y = y_df["target"].values

# 5-fold Stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.1,
        depth=6,
        class_weights=[20, 1],
        verbose=0,
        random_seed=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    score = amex_metric(
        pd.DataFrame({"target": y_val}), pd.DataFrame({"prediction": preds})
    )
    scores.append(score)

print(f"Mean AMEX metric: {np.mean(scores):.6f}")
