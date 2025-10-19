import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

# Target
y = np.log1p(train["SalePrice"])

# Features to drop
drop_cols = ["Id", "SalePrice"]
features = [c for c in train.columns if c not in drop_cols]

# Identify numerical and categorical
num_cols = train[features].select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = train[features].select_dtypes(include=["object"]).columns.tolist()

# Impute numerical
medians = train[num_cols].median()
train[num_cols] = train[num_cols].fillna(medians)
test[num_cols] = test[num_cols].fillna(medians)

# Impute categorical
train[cat_cols] = train[cat_cols].fillna("MISSING")
test[cat_cols] = test[cat_cols].fillna("MISSING")

# Encode categoricals with LabelEncoder fitted on both
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

X = train[features]
X_test = test[features]

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []
for train_idx, valid_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

    model = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        early_stopping_rounds=50,
        verbose=False,
    )
    pred_val = model.predict(X_val, num_iteration=model.best_iteration_)
    rmse = np.sqrt(mean_squared_error(y_val, pred_val))
    rmse_scores.append(rmse)

cv_rmse = np.mean(rmse_scores)
print(f"CV RMSE (log-scale): {cv_rmse:.5f}")

# Retrain on full data
final_model = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42
)
final_model.fit(X, y, verbose=False)

# Predict and write submission
pred_test_log = final_model.predict(X_test)
pred_test = np.expm1(pred_test_log)
submission = pd.DataFrame({"Id": test["Id"], "SalePrice": pred_test})
os.makedirs("./working", exist_ok=True)
submission.to_csv("./working/submission.csv", index=False)
