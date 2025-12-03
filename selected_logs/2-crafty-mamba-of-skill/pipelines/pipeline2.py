import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Load and concatenate data
files = ["./input/train_0.csv", "./input/train_1.csv", "./input/train_2.csv"]
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
del dfs

# Targets and grouping
targets = ["team_A_scoring_within_10sec", "team_B_scoring_within_10sec"]
group = df["game_num"]

# Prepare feature matrix
drop_cols = [
    "game_num",
    "event_id",
    "player_scoring_next",
    "team_scoring_next",
] + targets
features = [c for c in df.columns if c not in drop_cols]
X = df[features].astype("float32")
y = df[targets].astype("int8")

# 5-fold GroupKFold by game_num
gkf = GroupKFold(n_splits=5)
oof_preds = np.zeros_like(y.values, dtype=float)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=group)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    for i, target in enumerate(targets):
        train_data = lgb.Dataset(X_tr, label=y_tr[target])
        val_data = lgb.Dataset(X_val, label=y_val[target], reference=train_data)
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.1,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbosity": -1,
        }
        bst = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[val_data],
            early_stopping_rounds=10,
            verbose_eval=False,
        )
        oof_preds[val_idx, i] = bst.predict(X_val, num_iteration=bst.best_iteration)

# Compute AUCs
auc_A = roc_auc_score(y["team_A_scoring_within_10sec"], oof_preds[:, 0])
auc_B = roc_auc_score(y["team_B_scoring_within_10sec"], oof_preds[:, 1])
mean_auc = 0.5 * (auc_A + auc_B)
print(f"AUC Team A: {auc_A:.4f}, AUC Team B: {auc_B:.4f}, Mean AUC: {mean_auc:.4f}")
