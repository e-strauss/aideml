import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
test_ids = test["Id"]

# Target
y = np.log1p(train["SalePrice"])

# Drop id and target
X = train.drop(["Id", "SalePrice"], axis=1).copy()
X_test = test.drop("Id", axis=1).copy()

# Feature engineering: TotalSF
X["TotalBsmtSF"] = X["TotalBsmtSF"].fillna(0)
X_test["TotalBsmtSF"] = X_test["TotalBsmtSF"].fillna(0)
X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
X_test["TotalSF"] = X_test["TotalBsmtSF"] + X_test["1stFlrSF"] + X_test["2ndFlrSF"]

# Feature engineering: TotalBath = full baths + 0.5*half baths (basement + above)
for df in [X, X_test]:
    df["FullBath"] = df["FullBath"].fillna(0)
    df["HalfBath"] = df["HalfBath"].fillna(0)
    df["BsmtFullBath"] = df["BsmtFullBath"].fillna(0)
    df["BsmtHalfBath"] = df["BsmtHalfBath"].fillna(0)
    df["TotalBath"] = (
        df["FullBath"]
        + df["BsmtFullBath"]
        + 0.5 * (df["HalfBath"] + df["BsmtHalfBath"])
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
    model = LGBMRegressor(random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    oof_preds[val_idx] = model.predict(X_val)

# Evaluate
rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"CV RMSE: {rmse:.5f}")

# Train full model and predict test
final_model = LGBMRegressor(random_state=42, n_jobs=-1)
final_model.fit(X, y)
test_preds = final_model.predict(X_test)
submission = pd.DataFrame({"Id": test_ids, "SalePrice": np.expm1(test_preds)})
submission.to_csv("./working/submission.csv", index=False)
