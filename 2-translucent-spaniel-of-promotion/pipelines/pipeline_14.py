import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
test_ids = test["Id"]

# Prepare features and target
y = np.log1p(train["SalePrice"])
X = train.drop(["Id", "SalePrice"], axis=1).copy()
X_test = test.drop("Id", axis=1).copy()

# Existing feature engineering
# TotalSF
X["TotalBsmtSF"] = X["TotalBsmtSF"].fillna(0)
X_test["TotalBsmtSF"] = X_test["TotalBsmtSF"].fillna(0)
X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
X_test["TotalSF"] = X_test["TotalBsmtSF"] + X_test["1stFlrSF"] + X_test["2ndFlrSF"]

# Age-based features
X["GarageYrBlt"] = X["GarageYrBlt"].fillna(X["YrSold"])
X_test["GarageYrBlt"] = X_test["GarageYrBlt"].fillna(X_test["YrSold"])
X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
X_test["HouseAge"] = X_test["YrSold"] - X_test["YearBuilt"]
X["SinceRemodel"] = X["YrSold"] - X["YearRemodAdd"]
X_test["SinceRemodel"] = X_test["YrSold"] - X_test["YearRemodAdd"]
X["GarageAge"] = X["YrSold"] - X["GarageYrBlt"]
X_test["GarageAge"] = X_test["YrSold"] - X_test["GarageYrBlt"]

# Porches
porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
X["TotalPorchSF"] = X[porch_cols].sum(axis=1)
X_test["TotalPorchSF"] = X_test[porch_cols].sum(axis=1)

# New feature: TotalBath
# Fill missing basement bath counts as zero
for col in ["BsmtFullBath", "BsmtHalfBath"]:
    X[col] = X[col].fillna(0)
    X_test[col] = X_test[col].fillna(0)
X["TotalBath"] = (
    X["FullBath"] + 0.5 * X["HalfBath"] + X["BsmtFullBath"] + 0.5 * X["BsmtHalfBath"]
)
X_test["TotalBath"] = (
    X_test["FullBath"]
    + 0.5 * X_test["HalfBath"]
    + X_test["BsmtFullBath"]
    + 0.5 * X_test["BsmtHalfBath"]
)

# Preprocessing: median impute numeric, label encode categoricals
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].fillna("Missing").astype("category").cat.codes
        X_test[col] = X_test[col].fillna("Missing").astype("category").cat.codes
    else:
        med = X[col].median()
        X[col] = X[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

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
