import skrub
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, make_scorer
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
test_ids = test["Id"]

# Start skrub DataOps plan
data = skrub.var("data", train).skb.subsample(n=100)

# Target: log1p transform
y = data["SalePrice"].skb.mark_as_y()
y_log = y.skb.apply_func(np.log1p)

# Features: drop Id and SalePrice, mark as X
X = data.drop(["Id", "SalePrice"], axis=1).skb.mark_as_X()

# Feature engineering: fillna TotalBsmtSF as 0, then create TotalSF
X_bsmt = X.assign(TotalBsmtSF=X["TotalBsmtSF"].skb.apply_func(lambda s: s.fillna(0)))
X_total_sf = X_bsmt.assign(TotalSF=X_bsmt["TotalBsmtSF"] + X_bsmt["1stFlrSF"] + X_bsmt["2ndFlrSF"])

# GarageYrBlt fillna with YrSold
X_garageyr = X_total_sf.assign(GarageYrBlt=X_total_sf["GarageYrBlt"].skb.apply_func(
    lambda s: s.fillna(X_total_sf["YrSold"])
))

# Age-based features
X_house_age = X_garageyr.assign(HouseAge=X_garageyr["YrSold"] - X_garageyr["YearBuilt"])
X_since_remodel = X_house_age.assign(SinceRemodel=X_house_age["YrSold"] - X_house_age["YearRemodAdd"])
X_garage_age = X_since_remodel.assign(GarageAge=X_since_remodel["YrSold"] - X_since_remodel["GarageYrBlt"])

# Preprocessing: median impute numeric, label encode categoricals
obj_selector = skrub.selectors.filter(lambda col: col.dtype == "object")
num_selector = skrub.selectors.filter(lambda col: col.dtype != "object")

X_obj = X_garage_age.skb.select(obj_selector)
X_num = X_garage_age.skb.select(num_selector)

# Object/categorical: fillna "Missing", then OrdinalEncoder
X_obj_imp = X_obj.skb.apply(SimpleImputer(strategy="constant", fill_value="Missing"))
X_obj_enc = X_obj_imp.skb.apply(OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))

# Numeric: median impute
X_num_imp = X_num.skb.apply(SimpleImputer(strategy="median"))

# Concatenate all features
X_final = X_num_imp.skb.concat([X_obj_enc], axis=1)

# Model
model = LGBMRegressor(random_state=42)
pred_log = X_final.skb.apply(model, y=y_log)

# Inverse log1p for predictions (only in predict mode)
mode = skrub.eval_mode()
inv_func = mode.skb.match({"predict": np.expm1}, default=(lambda x: x))
pred = pred_log.skb.apply_func(inv_func)

# Cross-validation
learner = pred.skb.make_learner()
data_ = pred.skb.get_data()
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