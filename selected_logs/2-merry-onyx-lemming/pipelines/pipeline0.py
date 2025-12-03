import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
import lightgbm as lgb

# Load and concatenate train data
files = [
    os.path.join("input", f)
    for f in os.listdir("input")
    if f.startswith("train_") and f.endswith(".csv")
]
df_list = []
for f in files:
    df_list.append(pd.read_csv(f))
df = pd.concat(df_list, ignore_index=True)

# Define features and targets
drop_cols = [
    "game_num",
    "event_id",
    "player_scoring_next",
    "team_scoring_next",
    "team_A_scoring_within_10sec",
    "team_B_scoring_within_10sec",
]
features = [c for c in df.columns if c not in drop_cols]
X = df[features].astype(np.float32)
yA = df["team_A_scoring_within_10sec"].fillna(0).astype(int).values
yB = df["team_B_scoring_within_10sec"].fillna(0).astype(int).values
groups = df["game_num"].values

# Prepare CV
gkf = GroupKFold(n_splits=5)
predsA = np.zeros(len(df), dtype=np.float32)
predsB = np.zeros(len(df), dtype=np.float32)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "seed": 42,
}

for fold, (trn_idx, val_idx) in enumerate(gkf.split(X, yA, groups)):
    X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    yA_tr, yA_val = yA[trn_idx], yA[val_idx]
    yB_tr, yB_val = yB[trn_idx], yB[val_idx]

    # Team A model
    dtrainA = lgb.Dataset(X_tr, yA_tr)
    dvalA = lgb.Dataset(X_val, yA_val, reference=dtrainA)
    modelA = lgb.train(
        params,
        dtrainA,
        num_boost_round=1000,
        valid_sets=[dvalA],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    predsA[val_idx] = modelA.predict(X_val, num_iteration=modelA.best_iteration)

    # Team B model
    dtrainB = lgb.Dataset(X_tr, yB_tr)
    dvalB = lgb.Dataset(X_val, yB_val, reference=dtrainB)
    modelB = lgb.train(
        params,
        dtrainB,
        num_boost_round=1000,
        valid_sets=[dvalB],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    predsB[val_idx] = modelB.predict(X_val, num_iteration=modelB.best_iteration)

    print(f"Fold {fold+1} done.")

# Compute log loss
llA = log_loss(yA, predsA)
llB = log_loss(yB, predsB)
print(f"Log Loss Team A: {llA:.6f}")
print(f"Log Loss Team B: {llB:.6f}")
print(f"Average Log Loss: {((llA+llB)/2):.6f}")
