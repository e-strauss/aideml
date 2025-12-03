import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

# Load and concatenate training data
files = ["./input/train_0.csv", "./input/train_1.csv", "./input/train_2.csv"]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Targets
yA = df["team_A_scoring_within_10sec"]
yB = df["team_B_scoring_within_10sec"]

# Features: drop identifiers and other targets
drop_cols = [
    "game_num",
    "event_id",
    "event_time",
    "player_scoring_next",
    "team_scoring_next",
    "team_A_scoring_within_10sec",
    "team_B_scoring_within_10sec",
]
X = df.drop(columns=drop_cols).fillna(-1)
groups = df["event_id"].values

# Prepare for out-of-fold predictions
n = len(df)
oofA = np.zeros(n)
oofB = np.zeros(n)

# 5-fold GroupKFold by event_id
gkf = GroupKFold(n_splits=5)
for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, yA, groups)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    yA_tr, yA_val = yA.iloc[tr_idx], yA.iloc[val_idx]
    yB_tr, yB_val = yB.iloc[tr_idx], yB.iloc[val_idx]

    # Team A model
    modelA = LGBMClassifier(
        objective="binary", n_estimators=100, random_state=42, n_jobs=-1
    )
    modelA.fit(
        X_tr,
        yA_tr,
        eval_set=[(X_val, yA_val)],
        eval_metric="auc",
        early_stopping_rounds=10,
        verbose=False,
    )
    oofA[val_idx] = modelA.predict_proba(X_val)[:, 1]

    # Team B model
    modelB = LGBMClassifier(
        objective="binary", n_estimators=100, random_state=42, n_jobs=-1
    )
    modelB.fit(
        X_tr,
        yB_tr,
        eval_set=[(X_val, yB_val)],
        eval_metric="auc",
        early_stopping_rounds=10,
        verbose=False,
    )
    oofB[val_idx] = modelB.predict_proba(X_val)[:, 1]

# Compute and print ROC AUC scores
aucA = roc_auc_score(yA, oofA)
aucB = roc_auc_score(yB, oofB)
print(f"CV AUC Team A: {aucA:.6f}")
print(f"CV AUC Team B: {aucB:.6f}")
print(f"Mean CV AUC: {(aucA + aucB) / 2:.6f}")
