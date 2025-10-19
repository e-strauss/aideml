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

# Feature engineering
for df in [X, X_test]:
    # TotalSF
    df["TotalBsmtSF"] = df["TotalBsmtSF"].fillna(0)
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    # Age features
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YrSold"])
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["SinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]
    df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
    # Porch
    porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
    df["TotalPorchSF"] = df[porch_cols].sum(axis=1)

# Ordinal quality mappings
qual_mapping = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
ordinal_cols = [
    "ExterQual",
    "ExterCond",
    "HeatingQC",
    "KitchenQual",
    "FireplaceQu",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
]
for col in ordinal_cols:
    X[col] = X[col].map(qual_mapping).fillna(0)
    X_test[col] = X_test[col].map(qual_mapping).fillna(0)

# Preprocessing: median impute numeric, label encode categoricals
for col in X.columns:
    if col in ordinal_cols:  # already numeric
        continue
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
