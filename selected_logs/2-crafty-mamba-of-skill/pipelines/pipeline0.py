import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
import lightgbm as lgb

# 1. Load and concatenate data
files = [
    os.path.join("input", fname)
    for fname in ["train_0.csv", "train_1.csv", "train_2.csv"]
]
df_list = []
for f in files:
    df = pd.read_csv(f)
    # downcast to save memory
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = df[c].astype("int32")
    df_list.append(df)
data = pd.concat(df_list, ignore_index=True)
del df_list

# 2. Prepare features and targets
target_cols = ["team_A_scoring_within_10sec", "team_B_scoring_within_10sec"]
drop_cols = [
    "game_num",
    "event_id",
    "player_scoring_next",
    "team_scoring_next",
] + target_cols
features = [c for c in data.columns if c not in drop_cols]
X = data[features]
yA = data["team_A_scoring_within_10sec"].values
yB = data["team_B_scoring_within_10sec"].values
groups = data["game_num"].values
del data

# 3. Cross-validation
gkf = GroupKFold(n_splits=5)
loglosses = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, groups=groups)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    yA_tr, yA_val = yA[train_idx], yA[val_idx]
    yB_tr, yB_val = yB[train_idx], yB[val_idx]

    # Model for Team A
    modelA = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=100,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
    )
    modelA.fit(
        X_tr,
        yA_tr,
        eval_set=[(X_val, yA_val)],
        eval_metric="binary_logloss",
        early_stopping_rounds=10,
        verbose=False,
    )
    pA = modelA.predict_proba(X_val)[:, 1]
    lossA = log_loss(yA_val, pA)

    # Model for Team B
    modelB = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=100,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
    )
    modelB.fit(
        X_tr,
        yB_tr,
        eval_set=[(X_val, yB_val)],
        eval_metric="binary_logloss",
        early_stopping_rounds=10,
        verbose=False,
    )
    pB = modelB.predict_proba(X_val)[:, 1]
    lossB = log_loss(yB_val, pB)

    loglosses.append((lossA + lossB) / 2)
    print(f"Fold {fold+1} log loss (avg of A&B): {(lossA + lossB)/2:.5f}")

# 4. Report mean log loss
mean_logloss = np.mean(loglosses)
print(f"\nMean log loss over 5 folds: {mean_logloss:.5f}")
