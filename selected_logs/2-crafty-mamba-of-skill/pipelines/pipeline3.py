import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Load and concatenate training data
files = ["./input/train_0.csv", "./input/train_1.csv", "./input/train_2.csv"]
dfs = [pd.read_csv(f) for f in files]
data = pd.concat(dfs, ignore_index=True)
del dfs

# Define features and targets
targets = ["team_A_scoring_within_10sec", "team_B_scoring_within_10sec"]
features = [
    c for c in data.columns if c not in ["game_num", "event_id", "event_time"] + targets
]
X = data[features]
y = data[targets]
groups = data["game_num"]
del data  # free memory

# Prepare cross-validation
n_splits = 5
gkf = GroupKFold(n_splits=n_splits)
auc_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    fold_aucs = []
    for target in targets:
        train_data = lgb.Dataset(X_train, label=y_train[target])
        val_data = lgb.Dataset(X_val, label=y_val[target], reference=train_data)
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbose": -1,
        }
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            early_stopping_rounds=20,
            verbose_eval=False,
        )
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        auc = roc_auc_score(y_val[target], preds)
        fold_aucs.append(auc)
    auc_scores.append(np.mean(fold_aucs))
    print(f"Fold {fold+1} AUC: {np.mean(fold_aucs):.6f}")

print(f"Mean CV AUC: {np.mean(auc_scores):.6f}")
