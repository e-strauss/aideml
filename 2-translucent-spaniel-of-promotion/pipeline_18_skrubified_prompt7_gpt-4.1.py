import pandas as pd
import numpy as np
import skrub
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# Load data
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")
test_ids = test["Id"]

# Skrub DataOps plan
data = skrub.var("data", train).skb.subsample(n=100)

# Target
y = data["SalePrice"].skb.mark_as_y()
y_log = y.skb.apply_func(np.log1p)

# Features
X = data.drop(["Id", "SalePrice"], axis=1).skb.mark_as_X()

# Impute LotFrontage by Neighborhood median, then global median
def lotfrontage_by_neigh(X):
    X = X.copy()
    lf_median = X.groupby("Neighborhood")["LotFrontage"].median()
    X["LotFrontage"] = X["LotFrontage"].fillna(X["Neighborhood"].map(lf_median))
    global_median_lf = X["LotFrontage"].median()
    X["LotFrontage"] = X["LotFrontage"].fillna(global_median_lf)
    return X

X_lf = X.skb.apply_func(lotfrontage_by_neigh)

# Feature engineering: TotalSF
X_bsmt = X_lf.assign(TotalBsmtSF=X_lf["TotalBsmtSF"].skb.apply_func(lambda x: x.fillna(0)))
X_totalsf = X_bsmt.assign(TotalSF=X_bsmt["TotalBsmtSF"] + X_bsmt["1stFlrSF"] + X_bsmt["2ndFlrSF"])

# Age features
X_garageyr = X_totalsf.assign(GarageYrBlt=X_totalsf["GarageYrBlt"].skb.apply_func(
    lambda col, df: col.fillna(df["YrSold"]), args=(X_totalsf,)
))
X_houseage = X_garageyr.assign(HouseAge=X_garageyr["YrSold"] - X_garageyr["YearBuilt"])
X_remodel = X_houseage.assign(SinceRemodel=X_houseage["YrSold"] - X_houseage["YearRemodAdd"])
X_garageage = X_remodel.assign(GarageAge=X_remodel["YrSold"] - X_remodel["GarageYrBlt"])

# Total porch square footage
porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
X_porch = X_garageage.assign(TotalPorchSF=X_garageage[porch_cols].skb.apply_func(lambda df: df.sum(axis=1)))

# Preprocessing: median impute numeric, label encode categorical
obj_selector = skrub.selectors.filter(lambda col: col.dtype == "object")
num_selector = skrub.selectors.filter(lambda col: col.dtype != "object")

X_obj = X_porch.skb.select(obj_selector)
X_num = X_porch.skb.select(num_selector)

# Categorical: fillna "Missing", then OrdinalEncoder
X_obj_imp = X_obj.skb.apply(SimpleImputer(strategy="constant", fill_value="Missing"))
X_obj_enc = X_obj_imp.skb.apply(OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))

# Numeric: median impute
X_num_imp = X_num.skb.apply(SimpleImputer(strategy="median"))

# Concatenate
X_final = X_num_imp.skb.concat([X_obj_enc], axis=1)

# Model
model = LGBMRegressor(random_state=42)
pred_log = X_final.skb.apply(model, y=y_log)

# Inverse log1p for predictions (only in predict mode)
mode = skrub.eval_mode()
function = mode.skb.match({"predict": np.expm1}, default=(lambda x: x))
pred = pred_log.skb.apply_func(function)

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

# Predict test
test_data = pd.read_csv("input/test.csv")
test_X = test_data.drop("Id", axis=1)
test_preds = learner.predict({"_skrub_X": test_X})

submission = pd.DataFrame({"Id": test_ids, "SalePrice": test_preds})
submission.to_csv("./working/submission_skrub.csv", index=False)