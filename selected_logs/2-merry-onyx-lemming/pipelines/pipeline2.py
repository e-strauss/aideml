import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

# Load and concatenate training data
files = ["./input/train_0.csv", "./input/train_1.csv", "./input/train_2.csv"]
dfs = [pd.read_csv(f, low_memory=False) for f in files]
df = pd.concat(dfs, ignore_index=True)

# Define targets and drop unused columns
target_cols = ["team_A_scoring_within_10sec", "team_B_scoring_within_10sec"]
drop_cols = ["event_id", "player_scoring_next", "team_scoring_next"]
features = [c for c in df.columns if c not in target_cols + drop_cols + ["game_num"]]

X = df[features].values
yA = df["team_A_scoring_within_10sec"].values
yB = df["team_B_scoring_within_10sec"].values
groups = df["game_num"].values

# Impute missing values with column means
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Prepare cross-validation
kf = GroupKFold(n_splits=5)
oof_preds_A = np.zeros_like(yA, dtype=float)
oof_preds_B = np.zeros_like(yB, dtype=float)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, yA, groups)):
    X_train, X_val = X[train_idx], X[val_idx]
    yA_train, yA_val = yA[train_idx], yA[val_idx]
    yB_train, yB_val = yB[train_idx], yB[val_idx]

    # Model for Team A
    modelA = XGBClassifier(
        n_estimators=500,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    modelA.fit(
        X_train,
        yA_train,
        eval_set=[(X_val, yA_val)],
        early_stopping_rounds=30,
        verbose=False,
    )
    oof_preds_A[val_idx] = modelA.predict_proba(X_val)[:, 1]

    # Model for Team B
    modelB = XGBClassifier(
        n_estimators=500,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    modelB.fit(
        X_train,
        yB_train,
        eval_set=[(X_val, yB_val)],
        early_stopping_rounds=30,
        verbose=False,
    )
    oof_preds_B[val_idx] = modelB.predict_proba(X_val)[:, 1]

# Compute average log loss
lossA = log_loss(yA, oof_preds_A)
lossB = log_loss(yB, oof_preds_B)
cv_logloss = (lossA + lossB) / 2

print(f"CV log loss: {cv_logloss:.5f}")
