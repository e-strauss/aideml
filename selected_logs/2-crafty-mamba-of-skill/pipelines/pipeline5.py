import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Load data
files = ["./input/train_0.csv", "./input/train_1.csv", "./input/train_2.csv"]
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)
del dfs

# Targets and grouping
targets = ["team_A_scoring_within_10sec", "team_B_scoring_within_10sec"]
group = df["game_num"]

# Prepare features
drop_cols = [
    "game_num",
    "event_id",
    "player_scoring_next",
    "team_scoring_next",
] + targets
features = [c for c in df.columns if c not in drop_cols]
X = df[features].astype("float32")
y = df[targets].astype("int8")

# Prepare OOF predictions
oof_preds = np.zeros((len(df), len(targets)), dtype=float)

# 5-fold CV
gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=group)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    for i, target in enumerate(targets):
        model = lgb.LGBMClassifier(
            objective="binary",
            learning_rate=0.1,
            n_estimators=100,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_tr,
            y_tr[target],
            eval_set=[(X_val, y_val[target])],
            eval_metric="auc",
            early_stopping_rounds=10,
            verbose=False,
        )
        oof_preds[val_idx, i] = model.predict_proba(X_val)[:, 1]

# Compute and print AUCs
auc_A = roc_auc_score(y[targets[0]], oof_preds[:, 0])
auc_B = roc_auc_score(y[targets[1]], oof_preds[:, 1])
mean_auc = 0.5 * (auc_A + auc_B)
print(f"AUC Team A: {auc_A:.4f}, AUC Team B: {auc_B:.4f}, Mean AUC: {mean_auc:.4f}")
