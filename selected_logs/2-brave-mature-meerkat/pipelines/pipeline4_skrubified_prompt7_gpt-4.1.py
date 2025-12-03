import skrub
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

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

# Skrub variable tracking
data_var = skrub.var("data", data)
labels_var = skrub.var("labels", labels)

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

# Sort and compute EWM features
df_sorted = data_var.sort_values(["customer_ID", "S_2"])
# EWM transform for all features
def ewm_group(df):
    df = df.copy()
    df[features] = df.groupby("customer_ID")[features].transform(
        lambda x: x.ewm(alpha=0.3).mean()
    )
    return df

df_ewm = df_sorted.skb.apply_func(ewm_group)

# Extract last EWM value per customer
def last_per_customer(df):
    return df.groupby("customer_ID").tail(1).reset_index(drop=True)

df_last = df_ewm.skb.apply_func(last_per_customer)

# X and y definition
X = df_last[features].skb.mark_as_X()
# Align y with X by customer_ID
def align_y(df_last, labels):
    y_df = (
        labels.set_index("customer_ID").loc[df_last["customer_ID"]].reset_index(drop=True)
    )
    return y_df

y_df = df_last.skb.apply_func(lambda df: align_y(df, labels_var))
y = y_df["target"].skb.mark_as_y()

# Model
model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=6,
    class_weights=[20, 1],
    verbose=0,
    random_seed=42,
)

pred = X.skb.apply(model, y=y)

# Skrub learner
learner = pred.skb.make_learner()

# Get data for cross-validation
data_ = pred.skb.get_data()

# Custom scorer for skrub cross_validate
from sklearn.metrics import make_scorer

def amex_metric_scorer(y_true, y_pred):
    return amex_metric(
        pd.DataFrame({"target": y_true}),
        pd.DataFrame({"prediction": y_pred}),
    )

scorer = make_scorer(amex_metric_scorer, needs_proba=True)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = skrub.cross_validate(
    learner,
    data_,
    cv=cv,
    scoring=scorer,
    return_train_score=False,
)

print(f"Mean AMEX metric: {np.mean(scores['test_score']):.6f}")