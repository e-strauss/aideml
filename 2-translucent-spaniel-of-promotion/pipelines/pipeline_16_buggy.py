import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

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

# XGBoost parameters
params = {
    "objective": "reg:squarederror",
    "eta": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
    "eval_metric": "rmse",
}

# 5-fold CV with native API
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmses = []
best_rounds = []

for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)

    evallist = [(dval, "eval")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evallist,
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    best_it = model.best_iteration
    best_rounds.append(best_it)

    preds_val = model.predict(dval, ntree_limit=best_it)
    rmse = np.sqrt(mean_squared_error(y_val, preds_val))
    rmses.append(rmse)

# Report CV score
cv_score = np.mean(rmses)
print(f"CV RMSE (log scale): {cv_score:.5f}")

# Retrain on full data
avg_best_it = int(np.mean(best_rounds) * 1.1)
dfull = xgb.DMatrix(X, label=y.values)
final_model = xgb.train(params, dfull, num_boost_round=avg_best_it, verbose_eval=False)

# Predict and save submission
dtest = xgb.DMatrix(X_test)
preds_test_log = final_model.predict(dtest, ntree_limit=avg_best_it)
preds_test = np.expm1(preds_test_log)

submission = pd.DataFrame({"Id": test_ids, "SalePrice": preds_test})
submission.to_csv("./working/submission.csv", index=False)
