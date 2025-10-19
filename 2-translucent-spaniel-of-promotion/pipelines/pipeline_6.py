import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
test_ids = test_df["Id"]

# Identify feature types
cat_feats = train_df.select_dtypes(include=["object"]).columns.tolist()
numeric_feats = [
    c for c in train_df.columns if c not in cat_feats + ["Id", "SalePrice"]
]

# Prepare features and target
y = np.log1p(train_df["SalePrice"])
X = train_df.drop(["Id", "SalePrice"], axis=1).copy()
X_test = test_df.drop("Id", axis=1).copy()

# Simple preprocessing: impute and encode
for col in numeric_feats:
    med = X[col].median()
    X[col] = X[col].fillna(med)
    X_test[col] = X_test[col].fillna(med)
for col in cat_feats:
    X[col] = X[col].fillna("Missing").astype("category").cat.codes
    X_test[col] = X_test[col].fillna("Missing").astype("category").cat.codes

# Log-transform highly skewed numeric features
skewness = X[numeric_feats].skew()
skewed_feats = skewness[skewness.abs() > 0.75].index.tolist()
for col in skewed_feats:
    X[col] = np.log1p(X[col])
    X_test[col] = np.log1p(X_test[col])

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model = LGBMRegressor(random_state=42)
    model.fit(X_tr, y_tr)
    oof_preds[val_idx] = model.predict(X_val)

# Evaluate
rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"CV RMSE: {rmse:.5f}")

# Train full model and predict test
final_model = LGBMRegressor(random_state=42)
final_model.fit(X, y)
test_preds = final_model.predict(X_test)
submission = pd.DataFrame({"Id": test_ids, "SalePrice": np.expm1(test_preds)})
submission.to_csv("./working/submission.csv", index=False)
