import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
train_ids = train["Id"]
test_ids = test["Id"]

# Separate target
y = np.log1p(train["SalePrice"])

# Combine for preprocessing
train_feat = train.drop(["Id", "SalePrice"], axis=1)
test_feat = test.drop(["Id"], axis=1)
all_feat = pd.concat([train_feat, test_feat], axis=0).reset_index(drop=True)

# Impute numeric features
num_cols = all_feat.select_dtypes(include=[np.number]).columns
medians = train_feat[num_cols].median()
all_feat[num_cols] = all_feat[num_cols].fillna(medians)

# Encode categorical features
cat_cols = all_feat.select_dtypes(include=["object"]).columns
for col in cat_cols:
    all_feat[col] = all_feat[col].fillna("NA").astype(str)
    all_feat[col], _ = pd.factorize(all_feat[col])

# Split back
X = all_feat.iloc[: len(train), :].values
X_test = all_feat.iloc[len(train) :, :].values

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmses = []
for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        early_stopping_rounds=50,
        verbose=False,
    )
    preds_val = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds_val))
    rmses.append(rmse)

# Report CV score
cv_score = np.mean(rmses)
print(f"CV RMSE (log scale): {cv_score:.5f}")

# Retrain on full data
final_model = XGBRegressor(
    n_estimators=int(model.best_iteration * 1.1),
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
final_model.fit(X, y, verbose=False)

# Predict and save submission
preds_test_log = final_model.predict(X_test)
preds_test = np.expm1(preds_test_log)
submission = pd.DataFrame({"Id": test_ids, "SalePrice": preds_test})
submission.to_csv("./working/submission.csv", index=False)
