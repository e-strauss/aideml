import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

# Load data
files = [
    os.path.join("input", f)
    for f in os.listdir("input")
    if f.startswith("train_") and f.endswith(".csv")
]
df = pd.concat([pd.read_csv(f) for f in sorted(files)], ignore_index=True)

# Define feature and target columns
drop_cols = [
    "game_num",
    "event_id",
    "event_time",
    "player_scoring_next",
    "team_scoring_next",
    "team_A_scoring_within_10sec",
    "team_B_scoring_within_10sec",
]
features = [c for c in df.columns if c not in drop_cols]
target_A = "team_A_scoring_within_10sec"
target_B = "team_B_scoring_within_10sec"

# Impute missing values
df[features] = df[features].fillna(-1)

X = df[features].values
yA = df[target_A].values
yB = df[target_B].values
groups = df["game_num"].values

# Prepare out-of-fold arrays
oof_pred_A = np.zeros_like(yA, dtype=float)
oof_pred_B = np.zeros_like(yB, dtype=float)

# 5-fold GroupKFold
kf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(kf.split(X, yA, groups)):
    print(f"Training fold {fold+1}/5...")
    X_train, X_val = X[train_idx], X[val_idx]
    yA_train, yA_val = yA[train_idx], yA[val_idx]
    yB_train, yB_val = yB[train_idx], yB[val_idx]

    # Model for Team A
    modelA = make_pipeline(
        StandardScaler(), LogisticRegression(solver="saga", penalty="l2", max_iter=100)
    )
    modelA.fit(X_train, yA_train)
    oof_pred_A[val_idx] = modelA.predict_proba(X_val)[:, 1]

    # Model for Team B
    modelB = make_pipeline(
        StandardScaler(), LogisticRegression(solver="saga", penalty="l2", max_iter=100)
    )
    modelB.fit(X_train, yB_train)
    oof_pred_B[val_idx] = modelB.predict_proba(X_val)[:, 1]

# Compute log losses
loss_A = log_loss(yA, oof_pred_A)
loss_B = log_loss(yB, oof_pred_B)
average_loss = 0.5 * (loss_A + loss_B)

print(f"CV log_loss Team A: {loss_A:.5f}")
print(f"CV log_loss Team B: {loss_B:.5f}")
print(f"Average CV log_loss: {average_loss:.5f}")
