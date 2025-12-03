import pandas as pd
import numpy as np
import skrub
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold

def amex_metric(y_true, y_pred):
    df = pd.concat([y_true, y_pred], axis=1)
    df = df.sort_values("prediction", ascending=False)
    df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
    four_pct_cutoff = int(0.04 * df["weight"].sum())
    df["weight_cumsum"] = df["weight"].cumsum()
    d = (df.loc[df["weight_cumsum"] <= four_pct_cutoff, "target"] == 1).sum() / (
        df["target"] == 1
    ).sum()
    df["random"] = (df["weight"] / df["weight"].sum()).cumsum()
    total_pos = (df["target"] * df["weight"]).sum()
    df["cum_pos_found"] = (df["target"] * df["weight"]).cumsum()
    df["lorentz"] = df["cum_pos_found"] / total_pos
    df["gini"] = (df["lorentz"] - df["random"]) * df["weight"]
    g = df["gini"].sum() / df["gini"].sum()
    return 0.5 * (g + d)

class CatBoostCatWrapper:
    def __init__(self, cat_features, **params):
        self.cat_features = cat_features
        self.params = params
        self.model = None
    def fit(self, X, y):
        pool = Pool(X, y, cat_features=self.cat_features)
        self.model = CatBoostClassifier(**self.params)
        self.model.fit(pool, eval_set=pool, verbose=False)
        return self
    def predict_proba(self, X):
        return self.model.predict_proba(X)

if __name__ == "__main__":
    # Load data
    data = pd.read_csv("./input/train_data_downsampled.csv")
    labels = pd.read_csv("./input/train_labels_downsampled.csv")
    # Skrub plan
    data_var = skrub.var("data", data)
    labels_var = skrub.var("labels", labels)
    # Take most recent statement per customer
    data_sorted = data_var.sort_values(["customer_ID", "S_2"])
    latest = data_sorted.drop_duplicates("customer_ID", keep="last").reset_index(drop=True)
    # Merge labels
    df = latest.merge(labels_var, on="customer_ID")
    # Prepare features
    cat_feats = [
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
    # Mark categorical columns
    for c in cat_feats:
        df[c] = df[c].astype("category")
    y = df["target"].skb.mark_as_y()
    X = df.drop(["customer_ID", "target"], axis=1).skb.mark_as_X()
    # CatBoost wrapper
    model = CatBoostCatWrapper(
        cat_features=cat_feats,
        iterations=200,
        learning_rate=0.1,
        depth=6,
        eval_metric="AUC",
        random_seed=42,
        class_weights=[20, 1],
        verbose=False,
        early_stopping_rounds=20,
    )
    pred = X.skb.apply(model, y=y)
    # StratifiedKFold
    splits = pred.skb.get_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    def amex_scorer(y_true, y_pred):
        y_true_df = pd.DataFrame({"target": y_true})
        y_pred_df = pd.DataFrame({"prediction": y_pred[:,1]})
        return amex_metric(y_true_df, y_pred_df)
    from sklearn.metrics import make_scorer
    scorer = make_scorer(amex_scorer, needs_proba=True, greater_is_better=True)
    learner = pred.skb.make_learner()
    scores = skrub.cross_validate(learner, splits, cv=cv, scoring=scorer, return_train_score=False)
    print(f"Mean CV score: {np.mean(scores['test_score']):.6f}")