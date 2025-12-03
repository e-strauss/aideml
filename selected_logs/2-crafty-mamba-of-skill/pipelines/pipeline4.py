import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

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
features = X.columns.tolist()

# Prepare for out-of-fold predictions
oofA = np.zeros(len(df))
oofB = np.zeros(len(df))

# 5-fold GroupKFold by event_id
gkf = GroupKFold(n_splits=5)
for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, yA, groups)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    yA_tr, yA_val = yA.iloc[tr_idx], yA.iloc[val_idx]
    yB_tr, yB_val = yB.iloc[tr_idx], yB.iloc[val_idx]

    for task, (y_tr, y_val, oof) in zip(
        ["A", "B"], [(yA_tr, yA_val, oofA), (yB_tr, yB_val, oofB)]
    ):
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "seed": 42,
        }
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[val_data],
            early_stopping_rounds=10,
            verbose_eval=False,
        )
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        oof[val_idx] = preds

# Compute and print ROC AUC scores
aucA = roc_auc_score(yA, oofA)
aucB = roc_auc_score(yB, oofB)
print(f"CV AUC Team A: {aucA:.6f}")
print(f"CV AUC Team B: {aucB:.6f}")
print(f"Mean CV AUC: {(aucA + aucB) / 2:.6f}")
