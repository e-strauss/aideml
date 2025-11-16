import skrub
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer

def amex_metric(y_true, y_pred):
    df = pd.DataFrame({"target": y_true, "prediction": y_pred})
    df = df.sort_values("prediction", ascending=False)
    df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
    four_pct_cutoff = int(0.04 * df["weight"].sum())
    df["weight_cumsum"] = df["weight"].cumsum()
    df_cut = df[df["weight_cumsum"] <= four_pct_cutoff]
    top4 = df_cut["target"].sum() / df["target"].sum()
    df["random"] = (df["weight"] / df["weight"].sum()).cumsum()
    total_pos = (df["target"] * df["weight"]).sum()
    df["cum_pos_found"] = (df["target"] * df["weight"]).cumsum()
    df["lorentz"] = df["cum_pos_found"] / total_pos
    df["gini"] = (df["lorentz"] - df["random"]) * df["weight"]
    weighted_gini = df["gini"].sum()
    df_true = df.copy()
    df_true["prediction"] = df_true["target"]
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
data = skrub.var("data", data)
data_sorted = data.sort_values(["customer_ID", "S_2"])
data_latest = data_sorted.drop_duplicates("customer_ID", keep="last")

labels = pd.read_csv("./input/train_labels_downsampled.csv")
labels = skrub.var("labels", labels)
df = data_latest.merge(labels, on="customer_ID", how="inner")

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

y = df["target"].skb.mark_as_y()
X = df.drop(columns=drop_cols + ["target"]).skb.mark_as_X()

model = lgb.LGBMClassifier(
    n_estimators=100, learning_rate=0.1, class_weight={0: 20, 1: 1}, random_state=42
)
pred = X.skb.apply(model, y=y)

# Use the amex_metric as a scorer, needs_proba=True to pass proba to metric
scorer = make_scorer(amex_metric, needs_proba=True)

# Prepare data for cross-validation
data_ = pred.skb.get_data()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
learner = pred.skb.make_learner()

scores = skrub.cross_validate(learner, data_, cv=cv, scoring=scorer, return_train_score=False)
print(f"Mean CV metric: {np.mean(scores['test_score']):.6f}")