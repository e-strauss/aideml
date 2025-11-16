import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb


# Competition metric
def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    def top_four_percent_captured(y_true, y_pred):
        df = pd.concat([y_true, y_pred], axis=1).sort_values(
            "prediction", ascending=False
        )
        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        cutoff = int(0.04 * df["weight"].sum())
        df["wcum"] = df["weight"].cumsum()
        return (df.loc[df["wcum"] <= cutoff, "target"] == 1).sum() / (
            df["target"] == 1
        ).sum()

    def weighted_gini(y_true, y_pred):
        df = pd.concat([y_true, y_pred], axis=1).sort_values(
            "prediction", ascending=False
        )
        df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
        df["rand"] = df["weight"].cumsum() / df["weight"].sum()
        df["cum_pos"] = (df["target"] * df["weight"]).cumsum()
        total_pos = (df["target"] * df["weight"]).sum()
        df["lorentz"] = df["cum_pos"] / total_pos
        df["gini"] = (df["lorentz"] - df["rand"]) * df["weight"]
        return df["gini"].sum()

    def norm_gini(y_true, y_pred):
        return weighted_gini(y_true, y_pred) / weighted_gini(
            y_true, y_true.rename(columns={"target": "prediction"})
        )

    y_t = y_true.reset_index(drop=True)
    y_p = y_pred.reset_index(drop=True)
    g = norm_gini(y_t, y_p)
    d = top_four_percent_captured(y_t, y_p)
    return 0.5 * (g + d)


# Load data
data = pd.read_csv("input/train_data_downsampled.csv")
labels = pd.read_csv("input/train_labels_downsampled.csv")

# Drop categorical/date, aggregate numeric features by mean
to_drop = [
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
    "S_2",
]
df = data.drop(columns=to_drop, errors="ignore")
df_agg = df.groupby("customer_ID").mean().reset_index()

# Merge with labels
df_all = df_agg.merge(labels, on="customer_ID", how="inner")
X = df_all.drop(["customer_ID", "target"], axis=1)
y = df_all["target"]
X.fillna(-999, inplace=True)

# 5-fold stratified CV
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for tr_idx, val_idx in kf.split(X, y):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=20,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_tr, y_tr)
    preds = model.predict_proba(X_val)[:, 1]
    y_val_df = pd.DataFrame({"target": y_val.reset_index(drop=True)})
    pred_df = pd.DataFrame({"prediction": preds})
    scores.append(amex_metric(y_val_df, pred_df))

print(f"CV mean amex metric: {np.mean(scores):.6f}")
