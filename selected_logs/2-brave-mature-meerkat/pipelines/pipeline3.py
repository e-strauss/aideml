import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb


def amex_metric(y_true, y_pred):
    # convert to DataFrame
    df = pd.DataFrame({"target": y_true, "prediction": y_pred})
    # weights
    df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
    # Gini
    df_g = df.sort_values("prediction", ascending=False).reset_index(drop=True)
    df_g["random"] = df_g["weight"].cumsum() / df_g["weight"].sum()
    total_pos = (df_g["target"] * df_g["weight"]).sum()
    df_g["cum_pos"] = (df_g["target"] * df_g["weight"]).cumsum()
    df_g["lorentz"] = df_g["cum_pos"] / total_pos
    df_g["gini_contrib"] = (df_g["lorentz"] - df_g["random"]) * df_g["weight"]
    weighted_gini = df_g["gini_contrib"].sum()
    # perfect Gini
    df_perfect = df_g.copy()
    df_perfect = df_perfect.sort_values("target", ascending=False).reset_index(
        drop=True
    )
    df_perfect["random"] = df_perfect["weight"].cumsum() / df_perfect["weight"].sum()
    df_perfect["cum_pos"] = (df_perfect["target"] * df_perfect["weight"]).cumsum()
    df_perfect["lorentz"] = df_perfect["cum_pos"] / total_pos
    df_perfect["gini_contrib"] = (
        df_perfect["lorentz"] - df_perfect["random"]
    ) * df_perfect["weight"]
    perfect_gini = df_perfect["gini_contrib"].sum()
    norm_gini = weighted_gini / perfect_gini
    # top4%
    df_d = df.sort_values("prediction", ascending=False).reset_index(drop=True)
    df_d["weight_cum"] = df_d["weight"].cumsum()
    cutoff = int(0.04 * df_d["weight"].sum())
    top = df_d.loc[df_d["weight_cum"] <= cutoff]
    d = top["target"].sum() / df_d["target"].sum()
    return 0.5 * (norm_gini + d)


# Load data
data = pd.read_csv("./input/train_data_downsampled.csv", low_memory=False)
labels = pd.read_csv("./input/train_labels_downsampled.csv")
# Drop categorical features
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
data = data.drop(columns=cat_cols)
# Aggregate numerical features per customer
num = data.select_dtypes(include=[np.number])
agg = num.groupby(data["customer_ID"]).agg(["mean", "std", "min", "max"])
# flatten columns
agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
agg = agg.reset_index()
# Merge with labels
df = labels.merge(agg, on="customer_ID", how="left")
X = df.drop(columns=["customer_ID", "target"])
y = df["target"].values
# Impute missing
X = X.fillna(-999)
# 5-fold stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    w_train = np.where(y_train == 0, 20, 1)
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train, sample_weight=w_train)
    y_pred = model.predict_proba(X_val)[:, 1]
    score = amex_metric(y_val, y_pred)
    scores.append(score)
    print(f"Fold AMEX metric: {score:.6f}")
print(f"Mean AMEX metric: {np.mean(scores):.6f}")
