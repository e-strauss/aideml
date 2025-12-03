import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
import lightgbm as lgb

# Load and concatenate training data
csv_files = glob.glob(os.path.join("input", "train_*.csv"))
df_list = []
for f in csv_files:
    df_list.append(pd.read_csv(f))
df = pd.concat(df_list, ignore_index=True)
# Cast types to save memory
for c in df.select_dtypes("float64").columns:
    df[c] = df[c].astype("float32")
for c in df.select_dtypes("int64").columns:
    df[c] = df[c].astype("int32")

# Define target columns and drop unused columns
target_cols = ["team_A_scoring_within_10sec", "team_B_scoring_within_10sec"]
drop_cols = ["event_id", "player_scoring_next", "team_scoring_next"]
features = [c for c in df.columns if c not in target_cols + drop_cols + ["game_num"]]

# Fill missing values
df[features] = df[features].fillna(-1)

# Prepare OOF storage
n = len(df)
oof_preds = {t: np.zeros(n, dtype=np.float32) for t in target_cols}

# 5-fold GroupKFold by game_num
gkf = GroupKFold(n_splits=5)
groups = df["game_num"].values

for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df[target_cols[0]], groups)):
    X_train, X_val = df.iloc[train_idx][features], df.iloc[val_idx][features]
    for t in target_cols:
        y_train = df.iloc[train_idx][t].values
        y_val = df.iloc[val_idx][t].values
        model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            early_stopping_rounds=10,
            verbose=False,
        )
        oof_preds[t][val_idx] = model.predict_proba(X_val)[:, 1]

# Compute log loss for each target
losses = []
for t in target_cols:
    losses.append(log_loss(df[t].values, oof_preds[t]))
# Average log loss
avg_loss = np.mean(losses)
print(f"CV Log Loss: {avg_loss:.5f}")
