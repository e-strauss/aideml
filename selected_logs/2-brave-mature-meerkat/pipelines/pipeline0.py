import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


# Competition metric
def amex_metric(y_true, y_pred):
    # combine
    df = pd.DataFrame({"target": y_true, "prediction": y_pred})
    df = df.sort_values("prediction", ascending=False)
    df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
    # top4%
    four_pct_cutoff = int(0.04 * df["weight"].sum())
    df["weight_cumsum"] = df["weight"].cumsum()
    df_cut = df[df["weight_cumsum"] <= four_pct_cutoff]
    top4 = df_cut["target"].sum() / df["target"].sum()
    # Gini
    df["random"] = (df["weight"] / df["weight"].sum()).cumsum()
    total_pos = (df["target"] * df["weight"]).sum()
    df["cum_pos_found"] = (df["target"] * df["weight"]).cumsum()
    df["lorentz"] = df["cum_pos_found"] / total_pos
    df["gini"] = (df["lorentz"] - df["random"]) * df["weight"]
    weighted_gini = df["gini"].sum()
    # normalized
    df_true = df.copy()
    df_true["prediction"] = df_true["target"]
    # perfect
    df_true = df_true.sort_values("prediction", ascending=False)
    df_true["weight"] = df_true["target"].apply(lambda x: 20 if x == 0 else 1)
    df_true["random"] = (df_true["weight"] / df_true["weight"].sum()).cumsum()
    total_pos2 = (df_true["target"] * df_true["weight"]).sum()
    df_true["cum_pos_found"] = (df_true["target"] * df_true["weight"]).cumsum()
    df_true["lorentz"] = df_true["cum_pos_found"] / total_pos2
    df_true["gini"] = (df_true["lorentz"] - df_true["random"]) * df_true["weight"]
    perfect_gini = df_true["gini"].sum()
    norm_gini = weighted_gini / perfect_gini
    return 0.5 * (norm_gini + top4)


# Load and aggregate latest snapshot per customer
data = pd.read_csv("./input/train_data_downsampled.csv", parse_dates=["S_2"])
data = data.sort_values(["customer_ID", "S_2"])
data = data.drop_duplicates("customer_ID", keep="last")

labels = pd.read_csv("./input/train_labels_downsampled.csv")
df = data.merge(labels, on="customer_ID", how="inner")

# Prepare features
drop_cols = [
    "customer_ID",
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
X = df.drop(columns=drop_cols + ["target"])
y = df["target"].values

# 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    model = lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.1, class_weight={0: 20, 1: 1}, random_state=42
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        early_stopping_rounds=10,
        verbose=False,
    )
    preds = model.predict_proba(X_val)[:, 1]
    score = amex_metric(y_val, preds)
    scores.append(score)
    print(f"Fold metric: {score:.6f}")

print(f"Mean CV metric: {np.mean(scores):.6f}")
