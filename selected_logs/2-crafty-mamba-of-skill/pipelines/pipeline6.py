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
fold_aucs = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    per_fold = []

    for target in targets:
        model = lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            learning_rate=0.05,
            num_leaves=31,
            n_estimators=500,
            verbose=-1,
        )
        model.fit(
            X_train,
            y_train[target],
            eval_set=[(X_val, y_val[target])],
            eval_metric="auc",
            early_stopping_rounds=20,
            verbose=False,
        )
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val[target], preds)
        per_fold.append(auc)

    mean_fold_auc = np.mean(per_fold)
    fold_aucs.append(mean_fold_auc)
    print(f"Fold {fold+1} AUC: {mean_fold_auc:.6f}")

print(f"Mean CV AUC: {np.mean(fold_aucs):.6f}")
