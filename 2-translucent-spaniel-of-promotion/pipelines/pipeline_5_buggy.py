import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

# Log-transform target
y = np.log1p(train["SalePrice"].values)

# Combine for consistent encoding
all_data = pd.concat([train.drop(columns=["SalePrice"]), test], sort=False).reset_index(
    drop=True
)

# Identify columns
id_col = "Id"
cat_cols = all_data.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in all_data.columns if c not in cat_cols + [id_col]]

# Impute numeric columns with median
for c in num_cols:
    median = all_data[c].median()
    all_data[c] = all_data[c].fillna(median)

# Label-encode categorical columns
for c in cat_cols:
    all_data[c] = all_data[c].fillna("NA")
    all_data[c], _ = pd.factorize(all_data[c])

# Split back into train and test sets
X = all_data.iloc[: train.shape[0], :].drop(columns=[id_col])
X_test = all_data.iloc[train.shape[0] :, :].drop(columns=[id_col])

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_list = []
for train_idx, valid_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_val = y[train_idx], y[valid_idx]
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        eval_metric="rmse",
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False,
    )
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    rmse_list.append(rmse)

cv_score = np.mean(rmse_list)
print(f"CV RMSE: {cv_score:.5f}")

# Train on full data
final_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
    eval_metric="rmse",
)
final_model.fit(X, y)

# Predict on test set and invert log
test_preds = final_model.predict(X_test)
test_preds = np.expm1(test_preds)

# Save submission
submission = pd.DataFrame({"Id": test[id_col], "SalePrice": test_preds})
submission.to_csv("./working/submission.csv", index=False)
