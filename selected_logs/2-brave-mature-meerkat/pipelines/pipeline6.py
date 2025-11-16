import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool


def top_four_percent_captured(y_true, y_pred):
    df = pd.concat([y_true, y_pred], axis=1).sort_values("prediction", ascending=False)
    df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
    cutoff = int(0.04 * df["weight"].sum())
    df["weight_cumsum"] = df["weight"].cumsum()
    top = df.loc[df["weight_cumsum"] <= cutoff]
    return (top["target"] == 1).sum() / (df["target"] == 1).sum()


def weighted_gini(y_true, y_pred):
    df = pd.concat([y_true, y_pred], axis=1).sort_values("prediction", ascending=False)
    df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
    df["random"] = (df["weight"] / df["weight"].sum()).cumsum()
    df["cum_pos_found"] = (df["target"] * df["weight"]).cumsum()
    total_pos = (df["target"] * df["weight"]).sum()
    df["lorentz"] = df["cum_pos_found"] / total_pos
    df["gini"] = (df["lorentz"] - df["random"]) * df["weight"]
    return df["gini"].sum()


def normalized_weighted_gini(y_true, y_pred):
    # Perfect model predictions = true labels
    y_true_copy = y_true.rename(columns={"target": "prediction"})
    return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_copy)


def amex_metric(y_true, y_pred):
    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)
    return 0.5 * (g + d)


if __name__ == "__main__":
    # Load and prepare data
    data = pd.read_csv("./input/train_data_downsampled.csv")
    labels = pd.read_csv("./input/train_labels_downsampled.csv")
    data = data.sort_values(["customer_ID", "S_2"])
    latest = data.drop_duplicates("customer_ID", keep="last").reset_index(drop=True)
    df = latest.merge(labels, on="customer_ID")
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
    y = df[["target"]].reset_index(drop=True)
    y_array = df["target"].values

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_array), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_array[train_idx], y_array[val_idx]
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
        preds = model.predict_proba(X_val)[:, 1]
        y_true_df = pd.DataFrame({"target": y_val})
        y_pred_df = pd.DataFrame({"prediction": preds})
        score = amex_metric(y_true_df, y_pred_df)
        scores.append(score)
        print(f"Fold {fold} score: {score:.6f}")
    print(f"Mean CV score: {np.mean(scores):.6f}")
