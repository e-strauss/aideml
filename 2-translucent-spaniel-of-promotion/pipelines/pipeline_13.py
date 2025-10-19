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

# Basic feature matrices
X = train.drop(["Id", "SalePrice"], axis=1).copy()
X_test = test.drop("Id", axis=1).copy()

# Feature engineering
for df in (X, X_test):
    df["TotalBsmtSF"] = df["TotalBsmtSF"].fillna(0)
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YrSold"])
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["SinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]
    df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
    porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
    df["TotalPorchSF"] = df[porch_cols].sum(axis=1)

# Separate Neighborhood for target encoding
neigh = X["Neighborhood"].copy()
neigh_test = X_test["Neighborhood"].copy()
X = X.drop("Neighborhood", axis=1)
X_test = X_test.drop("Neighborhood", axis=1)

# Preprocessing: median impute numeric, label encode other categoricals
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].fillna("Missing").astype("category").cat.codes
        X_test[col] = X_test[col].fillna("Missing").astype("category").cat.codes
    else:
        med = X[col].median()
        X[col] = X[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

# Prepare arrays for out-of-fold predictions and test encodings
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
neigh_te_test_folds = np.zeros((5, len(X_test)))

# CV with out-of-fold target encoding for Neighborhood
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    # mapping on train fold
    mapping = y_tr.groupby(neigh.iloc[train_idx]).mean()
    global_mean = y_tr.mean()
    # encode
    X_tr["Neighborhood_TE"] = neigh.iloc[train_idx].map(mapping).fillna(global_mean)
    X_val["Neighborhood_TE"] = neigh.iloc[val_idx].map(mapping).fillna(global_mean)
    neigh_te_test_folds[fold, :] = neigh_test.map(mapping).fillna(global_mean)
    # train
    model = LGBMRegressor(random_state=42)
    model.fit(X_tr, y_tr)
    oof_preds[val_idx] = model.predict(X_val)

# Evaluate
rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"CV RMSE: {rmse:.5f}")

# Prepare full-data encoding for final model
mapping_full = y.groupby(neigh).mean()
global_mean_full = y.mean()
X_full = X.copy()
X_full["Neighborhood_TE"] = neigh.map(mapping_full).fillna(global_mean_full)
X_test_full = X_test.copy()
X_test_full["Neighborhood_TE"] = neigh_test.map(mapping_full).fillna(global_mean_full)

# Train final model and predict test
final_model = LGBMRegressor(random_state=42)
final_model.fit(X_full, y)
test_preds = final_model.predict(X_test_full)
submission = pd.DataFrame({"Id": test_ids, "SalePrice": np.expm1(test_preds)})
submission.to_csv("./working/submission.csv", index=False)
