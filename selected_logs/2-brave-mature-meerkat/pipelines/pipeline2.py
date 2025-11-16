import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool
import warnings

warnings.filterwarnings("ignore")


def amex_metric(y_true, y_pred):
    df = pd.concat([y_true, y_pred], axis=1)
    df = df.sort_values("prediction", ascending=False)
    df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
    # Top Four Percent Captured
    four_pct_cutoff = int(0.04 * df["weight"].sum())
    df["weight_cumsum"] = df["weight"].cumsum()
    d = (df.loc[df["weight_cumsum"] <= four_pct_cutoff, "target"] == 1).sum() / (
        df["target"] == 1
    ).sum()
    # Weighted Gini
    df["random"] = (df["weight"] / df["weight"].sum()).cumsum()
    total_pos = (df["target"] * df["weight"]).sum()
    df["cum_pos_found"] = (df["target"] * df["weight"]).cumsum()
    df["lorentz"] = df["cum_pos_found"] / total_pos
    df["gini"] = (df["lorentz"] - df["random"]) * df["weight"]
    g = df["gini"].sum() / df["gini"].sum()  # normalized by itself
    return 0.5 * (g + d)


if __name__ == "__main__":
    # Load data
    data = pd.read_csv("./input/train_data_downsampled.csv")
    labels = pd.read_csv("./input/train_labels_downsampled.csv")
    # Take most recent statement per customer
    data_sorted = data.sort_values(["customer_ID", "S_2"])
    latest = data_sorted.drop_duplicates("customer_ID", keep="last").reset_index(
        drop=True
    )
    # Merge labels
    df = latest.merge(labels, on="customer_ID")
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
    for c in cat_feats:
        df[c] = df[c].astype("category")
    X = df.drop(["customer_ID", "target"], axis=1)
    y = df["target"].values
    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        train_pool = Pool(X_train, y_train, cat_features=cat_feats)
        val_pool = Pool(X_val, y_val, cat_features=cat_feats)
        model = CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            eval_metric="AUC",
            random_seed=42,
            class_weights=[20, 1],
            verbose=False,
            early_stopping_rounds=20,
        )
        model.fit(train_pool, eval_set=val_pool)
        pred = model.predict_proba(X_val)[:, 1]
        y_true_df = pd.DataFrame({"target": y_val})
        y_pred_df = pd.DataFrame({"prediction": pred})
        score = amex_metric(y_true_df, y_pred_df)
        scores.append(score)
        print(f"Fold score: {score:.6f}")
    print(f"Mean CV score: {np.mean(scores):.6f}")
