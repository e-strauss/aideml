import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
import lightgbm as lgb

# Load and concatenate data
dfs = [pd.read_csv(f"input/train_{i}.csv") for i in [0, 1, 2]]
df = pd.concat(dfs, ignore_index=True)

# Define targets and features
targets = ["team_A_scoring_within_10sec", "team_B_scoring_within_10sec"]
drop_cols = ["game_num", "event_id", "player_scoring_next", "team_scoring_next"]
features = [c for c in df.columns if c not in drop_cols + targets]

# Preprocessing
df[features] = df[features].fillna(0)
X = df[features].values
y = df[targets].values
groups = df["game_num"].values

# 5-fold GroupKFold CV
gkf = GroupKFold(n_splits=5)
fold_losses = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    preds = np.zeros_like(y_val, dtype=float)

    # Train one model per target
    for i in range(len(targets)):
        clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        clf.fit(
            X_train,
            y_train[:, i],
            eval_set=[(X_val, y_val[:, i])],
            early_stopping_rounds=10,
            verbose=False,
        )
        preds[:, i] = clf.predict_proba(X_val)[:, 1]

    loss = log_loss(y_val, preds)
    print(f"Fold {fold + 1} log loss: {loss:.5f}")
    fold_losses.append(loss)

print(f"Average log loss: {np.mean(fold_losses):.5f}")
