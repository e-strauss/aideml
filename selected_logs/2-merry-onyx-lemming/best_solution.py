import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
import lightgbm as lgb


def main():
    # Load and concatenate training data
    files = ["./input/train_0.csv", "./input/train_1.csv", "./input/train_2.csv"]
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    # Targets
    y = df[["team_A_scoring_within_10sec", "team_B_scoring_within_10sec"]].values
    # Features: drop identifiers and non-needed columns
    drop_cols = [
        "game_num",
        "event_id",
        "player_scoring_next",
        "team_scoring_next",
        "team_A_scoring_within_10sec",
        "team_B_scoring_within_10sec",
    ]
    X = df.drop(columns=drop_cols)
    # Impute missing values
    X = X.fillna(-1)
    # Prepare cross-validation
    groups = df["game_num"].values
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros_like(y, dtype=float)
    # Cross-validated training
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        for t in range(2):
            clf = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.1, random_state=42
            )
            clf.fit(
                X_train,
                y_train[:, t],
                eval_set=[(X_val, y_val[:, t])],
                early_stopping_rounds=20,
                verbose=False,
            )
            oof_preds[val_idx, t] = clf.predict_proba(X_val)[:, 1]
    # Compute log loss for each target and average
    loss_a = log_loss(y[:, 0], oof_preds[:, 0])
    loss_b = log_loss(y[:, 1], oof_preds[:, 1])
    avg_loss = 0.5 * (loss_a + loss_b)
    print(f"CV Log Loss Team A: {loss_a:.5f}")
    print(f"CV Log Loss Team B: {loss_b:.5f}")
    print(f"Average CV Log Loss: {avg_loss:.5f}")


if __name__ == "__main__":
    main()
