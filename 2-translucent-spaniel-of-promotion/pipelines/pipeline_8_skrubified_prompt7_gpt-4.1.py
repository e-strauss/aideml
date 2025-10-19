import pandas as pd
import numpy as np
import skrub
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, make_scorer
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
test_ids = test["Id"]

# Skrub variable for train data, with subsample for preview
data = skrub.var("data", train).skb.subsample(n=100)

# Target
y = data["SalePrice"].skb.apply_func(np.log1p).skb.mark_as_y()

# Drop id and target, mark as X
X = data.drop(["Id", "SalePrice"], axis=1).skb.mark_as_X()

# Feature engineering: TotalBsmtSF fillna(0)
X_totalbsmt = X.assign(TotalBsmtSF=X["TotalBsmtSF"].skb.apply_func(lambda s: s.fillna(0)))
# TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
X_totalsf = X_totalbsmt.assign(TotalSF=X_totalbsmt["TotalBsmtSF"] + X_totalbsmt["1stFlrSF"] + X_totalbsmt["2ndFlrSF"])

# Feature engineering: fillna(0) for bath columns
X_bath1 = X_totalsf.assign(FullBath=X_totalsf["FullBath"].skb.apply_func(lambda s: s.fillna(0)))
X_bath2 = X_bath1.assign(HalfBath=X_bath1["HalfBath"].skb.apply_func(lambda s: s.fillna(0)))
X_bath3 = X_bath2.assign(BsmtFullBath=X_bath2["BsmtFullBath"].skb.apply_func(lambda s: s.fillna(0)))
X_bath4 = X_bath3.assign(BsmtHalfBath=X_bath3["BsmtHalfBath"].skb.apply_func(lambda s: s.fillna(0)))
# TotalBath = FullBath + BsmtFullBath + 0.5*(HalfBath + BsmtHalfBath)
X_totalbath = X_bath4.assign(TotalBath=X_bath4["FullBath"] + X_bath4["BsmtFullBath"] + 0.5 * (X_bath4["HalfBath"] + X_bath4["BsmtHalfBath"]))

# Preprocessing: median impute numeric, label encode categoricals
obj_selector = skrub.selectors.filter(lambda col: col.dtype == "object")
num_selector = skrub.selectors.filter(lambda col: col.dtype != "object")

X_obj = X_totalbath.skb.select(obj_selector)
X_num = X_totalbath.skb.select(num_selector)

# Impute and encode categoricals
X_obj_imp = X_obj.skb.apply(SimpleImputer(strategy="constant", fill_value="Missing"))
X_obj_enc = X_obj_imp.skb.apply(OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))

# Impute numerics
X_num_imp = X_num.skb.apply(SimpleImputer(strategy="median"))

# Concatenate all features
X_final = X_num_imp.skb.concat([X_obj_enc], axis=1)

# Model
model = LGBMRegressor(random_state=42, n_jobs=-1)
pred = X_final.skb.apply(model, y=y)

# Inverse log1p for predictions (only in predict mode)
mode = skrub.eval_mode()
inv_func = mode.skb.match({"predict": np.expm1}, default=(lambda x: x))
pred_exp = pred.skb.apply_func(inv_func)

# Cross-validation
learner = pred_exp.skb.make_learner()
data_ = pred_exp.skb.get_data()
scorer = make_scorer(mean_squared_error)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = skrub.cross_validate(learner, data_, cv=cv, scoring=scorer, return_train_score=True)
rmse = np.sqrt(np.mean(scores["test_score"]))
print(f"CV RMSE: {rmse:.5f}")

# Train on full data
learner.fit(data_)

# Predict on test
test_data = test.drop("Id", axis=1)
test_preds = learner.predict({"_skrub_X": test_data})

submission = pd.DataFrame({"Id": test_ids, "SalePrice": test_preds})
submission.to_csv("./working/submission_skrub.csv", index=False)