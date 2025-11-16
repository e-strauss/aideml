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

# Define categorical columns to drop from numeric processing
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
# Identify numeric feature columns
features = [c for c in data.columns if c not in ["customer_ID"] + cat_cols]

# Sort and compute EWM features
df = data.sort_values(["customer_ID", "S_2"])
df[features] = df.groupby("customer_ID")[features].transform(
    lambda x: x.ewm(alpha=0.3).mean()
)

# Extract last EWM value per customer
df_last = df.groupby("customer_ID").tail(1).reset_index(drop=True)
X = df_last[features]
y_df = (
    labels.set_index("customer_ID").loc[df_last["customer_ID"]].reset_index(drop=True)
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
