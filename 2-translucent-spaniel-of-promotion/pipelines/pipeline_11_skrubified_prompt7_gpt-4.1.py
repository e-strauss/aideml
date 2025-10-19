import skrub
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.metrics import make_scorer

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
test_ids = test["Id"]

# Start DataOps plan
data = skrub.var("data", train).skb.subsample(n=100)

# Target
y = data["SalePrice"].skb.apply_func(np.log1p).skb.mark_as_y()

# Features
X = data.drop(["Id", "SalePrice"], axis=1).skb.mark_as_X()

# Feature engineering
X_bsmt = X.assign(TotalBsmtSF=X["TotalBsmtSF"].skb.apply_func(lambda s: s.fillna(0)))
X_total_sf = X_bsmt.assign(TotalSF=X_bsmt["TotalBsmtSF"] + X_bsmt["1stFlrSF"] + X_bsmt["2ndFlrSF"])

X_garage_yr = X_total_sf.assign(GarageYrBlt=X_total_sf["GarageYrBlt"].skb.apply_func(
    lambda s: s.fillna(X_total_sf["YrSold"])
))
X_house_age = X_garage_yr.assign(HouseAge=X_garage_yr["YrSold"] - X_garage_yr["YearBuilt"])
X_since_remodel = X_house_age.assign(SinceRemodel=X_house_age["YrSold"] - X_house_age["YearRemodAdd"])
X_garage_age = X_since_remodel.assign(GarageAge=X_since_remodel["YrSold"] - X_since_remodel["GarageYrBlt"])

porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
X_total_porch = X_garage_age.assign(
    TotalPorchSF=X_garage_age[porch_cols].skb.apply_func(lambda df: df.sum(axis=1))
)

# Preprocessing: median impute numeric, label encode categoricals
obj_selector = skrub.selectors.filter(lambda col: col.dtype == "object")
num_selector = skrub.selectors.filter(lambda col: col.dtype != "object")

X_obj = X_total_porch.skb.select(obj_selector)
X_num = X_total_porch.skb.select(num_selector)

X_obj_imp = X_obj.skb.apply(SimpleImputer(strategy="constant", fill_value="Missing"))
X_obj_enc = X_obj_imp.skb.apply(OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))

X_num_imp = X_num.skb.apply(SimpleImputer(strategy="median"))

X_proc = X_num_imp.skb.concat([X_obj_enc], axis=1)

# Model
model = LGBMRegressor(random_state=42)
pred_log = X_proc.skb.apply(model, y=y)

# Inverse log1p for predictions
mode = skrub.eval_mode()
inv_func = mode.skb.match({"predict": np.expm1}, default=(lambda x: x))
pred = pred_log.skb.apply_func(inv_func)

# CV
splits = pred.skb.get_data()
learner = pred.skb.make_learner()
scorer = make_scorer(mean_squared_error)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = skrub.cross_validate(learner, splits, cv=cv, scoring=scorer)
rmse = np.sqrt(np.mean(scores["test_score"]))
print(f"CV RMSE: {rmse:.5f}")

# Train on full data
learner.fit(splits)

# Predict test
test_data = test.drop("Id", axis=1)
test_preds = learner.predict({"_skrub_X": test_data})

submission = pd.DataFrame({"Id": test_ids, "SalePrice": test_preds})
submission.to_csv("./working/submission_skrub.csv", index=False)