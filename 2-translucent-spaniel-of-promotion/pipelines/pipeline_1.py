import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
test_ids = test["Id"]

# Separate target and drop Id
y = np.log1p(train["SalePrice"])
train = train.drop(["Id", "SalePrice"], axis=1)
test = test.drop(["Id"], axis=1)

# Identify numeric and categorical features
numeric_feats = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_feats = train.select_dtypes(include=["object"]).columns.tolist()

# Impute numeric features with median
for col in numeric_feats:
    median = train[col].median()
    train[col].fillna(median, inplace=True)
    test[col].fillna(median, inplace=True)

# Fill missing categoricals and one-hot encode
train[categorical_feats] = train[categorical_feats].fillna("Missing")
test[categorical_feats] = test[categorical_feats].fillna("Missing")
train_enc = pd.get_dummies(train, columns=categorical_feats)
test_enc = pd.get_dummies(test, columns=categorical_feats)

# Align train/test
train_enc, test_enc = train_enc.align(test_enc, join="left", axis=1, fill_value=0)

X = train_enc.values
X_test = test_enc.values

# 5-fold CV RMSE on log scale
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmses = []
for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    rmses.append(rmse)

print(f"CV RMSE (log scale): {np.mean(rmses):.5f}")

# Retrain on full data and predict test
final_model = Ridge(alpha=1.0, random_state=42)
final_model.fit(X, y)
test_preds_log = final_model.predict(X_test)
test_preds = np.expm1(test_preds_log)

# Save submission
submission = pd.DataFrame({"Id": test_ids, "SalePrice": test_preds})
submission.to_csv("./working/submission.csv", index=False)
