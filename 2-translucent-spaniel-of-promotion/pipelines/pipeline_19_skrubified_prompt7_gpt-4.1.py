import skrub
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, make_scorer

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
test_ids = test["Id"]

# Skrub variable for train data, subsample for preview
data = skrub.var("data", train).skb.subsample(n=100)

# Target
y = data["SalePrice"].skb.mark_as_y()
y_log = y.skb.apply_func(np.log1p)

# Features
X = data.drop(["Id", "SalePrice"], axis=1).skb.mark_as_X()

# Feature engineering
# TotalBsmtSF fillna(0)
X_totalbsmt = X.assign(TotalBsmtSF=X["TotalBsmtSF"].skb.apply_func(lambda s: s.fillna(0)))
# TotalSF
X_totalsf = X_totalbsmt.assign(TotalSF=X_totalbsmt["TotalBsmtSF"] + X_totalbsmt["1stFlrSF"] + X_totalbsmt["2ndFlrSF"])
# GarageYrBlt fillna(YrSold)
X_garageyr = X_totalsf.assign(GarageYrBlt=X_totalsf["GarageYrBlt"].skb.apply_func(lambda s: s.fillna(X_totalsf["YrSold"])))
# HouseAge
X_houseage = X_garageyr.assign(HouseAge=X_garageyr["YrSold"] - X_garageyr["YearBuilt"])
# SinceRemodel
X_remodel = X_houseage.assign(SinceRemodel=X_houseage["YrSold"] - X_houseage["YearRemodAdd"])
# GarageAge
X_garageage = X_remodel.assign(GarageAge=X_remodel["YrSold"] - X_remodel["GarageYrBlt"])
# TotalPorchSF
porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
X_porch = X_garageage.assign(TotalPorchSF=X_garageage[porch_cols].sum(axis=1))

# Ordinal quality mapping
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
# Map ordinal columns
def map_qual(col):
    return col.map(qual_mapping).fillna(0)
for col in ordinal_cols:
    X_porch = X_porch.assign(**{col: X_porch[col].skb.apply_func(map_qual)})

# Preprocessing: median impute numeric, label encode categoricals
def is_ordinal(col):
    return col.name in ordinal_cols
def is_object(col):
    return col.dtype == "object"
def is_numeric(col):
    return not is_object(col) and not is_ordinal(col)

ordinal_selector = skrub.selectors.filter_names(lambda name: name in ordinal_cols)
obj_selector = skrub.selectors.filter(lambda col: is_object(col) and not is_ordinal(col))
num_selector = skrub.selectors.filter(lambda col: is_numeric(col) and not is_ordinal(col))

X_ord = X_porch.skb.select(ordinal_selector)
# already numeric, nothing to do

X_obj = X_porch.skb.select(obj_selector)
# fillna("Missing") and label encode
X_obj_imp = X_obj.skb.apply(SimpleImputer(strategy="constant", fill_value="Missing"))
X_obj_enc = X_obj_imp.skb.apply(OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))

X_num = X_porch.skb.select(num_selector)
X_num_imp = X_num.skb.apply(SimpleImputer(strategy="median"))

# Concatenate all features
X_final = X_ord.skb.concat([X_obj_enc, X_num_imp], axis=1)

# Model
model = LGBMRegressor(random_state=42)
pred_log = X_final.skb.apply(model, y=y_log)

# Inverse log1p for predictions
mode = skrub.eval_mode()
inv_func = mode.skb.match({"predict": np.expm1}, default=(lambda x: x))
pred = pred_log.skb.apply_func(inv_func)

# CV
splits = pred.skb.get_data()
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
test_data = pd.read_csv("./input/test.csv")
test_X = test_data.drop("Id", axis=1)
test_preds = learner.predict({"_skrub_X": test_X})

submission = pd.DataFrame({"Id": test_data["Id"], "SalePrice": test_preds})
submission.to_csv("./working/submission_skrub.csv", index=False)