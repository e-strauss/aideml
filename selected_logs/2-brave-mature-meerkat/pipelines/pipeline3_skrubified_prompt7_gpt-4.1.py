import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

def amex_metric(y_true, y_pred):
    df = pd.DataFrame({"target": y_true, "prediction": y_pred})
    df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
    df_g = df.sort_values("prediction", ascending=False).reset_index(drop=True)
    df_g["random"] = df_g["weight"].cumsum() / df_g["weight"].sum()
    total_pos = (df_g["target"] * df_g["weight"]).sum()
    df_g["cum_pos"] = (df_g["target"] * df_g["weight"]).cumsum()
    df_g["lorentz"] = df_g["cum_pos"] / total_pos
    df_g["gini_contrib"] = (df_g["lorentz"] - df_g["random"]) * df_g["weight"]
    weighted_gini = df_g["gini_contrib"].sum()
    df_perfect = df_g.copy()
    df_perfect = df_perfect.sort_values("target", ascending=False).reset_index(drop=True)
    df_perfect["random"] = df_perfect["weight"].cumsum() / df_perfect["weight"].sum()
    df_perfect["cum_pos"] = (df_perfect["target"] * df_perfect["weight"]).cumsum()
    df_perfect["lorentz"] = df_perfect["cum_pos"] / total_pos
    df_perfect["gini_contrib"] = (df_perfect["lorentz"] - df_perfect["random"]) * df_perfect["weight"]
    perfect_gini = df_perfect["gini_contrib"].sum()
    norm_gini = weighted_gini / perfect_gini
    df_d = df.sort_values("prediction", ascending=False).reset_index(drop=True)
    df_d["weight_cum"] = df_d["weight"].cumsum()
    cutoff = int(0.04 * df_d["weight"].sum())
    top = df_d.loc[df_d["weight_cum"] <= cutoff]
    d = top["target"].sum() / df_d["target"].sum()
    return 0.5 * (norm_gini + d)

# Load data
data = pd.read_csv("./input/train_data_downsampled.csv", low_memory=False)
labels = pd.read_csv("./input/train_labels_downsampled.csv")

# Skrub plan
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
data_nocat = data_var.drop(columns=cat_cols)

# Aggregate numerical features per customer
num = data_nocat.select_dtypes(include=[np.number])
agg = num.groupby(data_nocat["customer_ID"]).agg(["mean", "std", "min", "max"])
agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
agg = agg.reset_index()

# Merge with labels
df = labels_var.merge(agg, on="customer_ID", how="left")

# Mark y and X
y = df["target"].skb.mark_as_y()
X = df.drop(columns=["customer_ID", "target"]).skb.mark_as_X()

# Impute missing
X_imp = X.fillna(-999)

# Model
model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42)

# Custom sample_weight function for training
def sample_weight_func(y):
    return np.where(y == 0, 20, 1)

# Skrub: pass sample_weight to model
pred = X_imp.skb.apply(model, y=y, fit_params={"sample_weight": y.skb.apply_func(sample_weight_func)})

# Make learner
learner = pred.skb.make_learner()

# Get data for cross-validation
data_ = pred.skb.get_data()

# Custom scorer for skrub
from sklearn.metrics import make_scorer
def amex_scorer(y_true, y_pred):
    return amex_metric(y_true, y_pred[:,1])
scorer = make_scorer(amex_scorer, needs_proba=True, greater_is_better=True)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = skrub.cross_validate(learner, data_, cv=cv, scoring=scorer, return_train_score=False)

for i, score in enumerate(scores["test_score"]):
    print(f"Fold AMEX metric: {score:.6f}")
print(f"Mean AMEX metric: {np.mean(scores['test_score']):.6f}")