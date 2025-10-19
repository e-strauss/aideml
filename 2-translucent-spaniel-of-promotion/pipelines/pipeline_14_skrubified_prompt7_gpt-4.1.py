import pandas as pd
import numpy as np
import skrub
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, make_scorer
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# Load data
data = skrub.var("data", pd.read_csv("./input/train.csv")).skb.subsample(n=100)

# Target
y = data["SalePrice"].skb.mark_as_y()
y_log = y.skb.apply_func(np.log1p)

# Features
X = data.drop(["Id", "SalePrice"], axis=1).skb.mark_as_X()

# Feature engineering
X_totalbsmt = X.assign(TotalBsmtSF=X["TotalBsmtSF"].skb.apply_func(lambda s: s.fillna(0)))
X_totalbsmt_test = None  # not used, test handled at predict time

X_totalsf = X_totalbsmt.assign(TotalSF=X_totalbsmt["TotalBsmtSF"] + X_totalbsmt["1stFlrSF"] + X_totalbsmt["2ndFlrSF"])

X_garageyrblt = X_totalsf.assign(GarageYrBlt=X_totalsf["GarageYrBlt"].skb.apply_func(
    lambda s: s.fillna(X_totalsf["YrSold"])
))
X_houseage = X_garageyrblt.assign(HouseAge=X_garageyrblt["YrSold"] - X_garageyrblt["YearBuilt"])
X_sinceremodel = X_houseage.assign(SinceRemodel=X_houseage["YrSold"] - X_houseage["YearRemodAdd"])
X_garageage = X_sinceremodel.assign(GarageAge=X_sinceremodel["YrSold"] - X_sinceremodel["GarageYrBlt"])

porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
X_totalporch = X_garageage.assign(
    TotalPorchSF=X_garageage[porch_cols].skb.apply_func(lambda df: df.sum(axis=1))
)

# Fill missing basement bath counts as zero
X_bsmtfullbath = X_totalporch.assign(BsmtFullBath=X_totalporch["BsmtFullBath"].skb.apply_func(lambda s: s.fillna(0)))
X_bsmthalfbath = X_bsmtfullbath.assign(BsmtHalfBath=X_bsmtfullbath["BsmtHalfBath"].skb.apply_func(lambda s: s.fillna(0)))

X_totalbath = X_bsmthalfbath.assign(
    TotalBath=(
        X_bsmthalfbath["FullBath"]
        + 0.5 * X_bsmthalfbath["HalfBath"]
        + X_bsmthalfbath["BsmtFullBath"]
        + 0.5 * X_bsmthalfbath["BsmtHalfBath"]
    )
)

# Preprocessing: median impute numeric, label encode categoricals
obj_selector = skrub.selectors.filter(lambda col: col.dtype == "object")
num_selector = skrub.selectors.filter(lambda col: col.dtype != "object")

X_obj = X_totalbath.skb.select(obj_selector)
X_num = X_totalbath.skb.select(num_selector)

X_obj_imp = X_obj.skb.apply(SimpleImputer(strategy="constant", fill_value="Missing"))
X_obj_enc = X_obj_imp.skb.apply(OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))

X_num_imp = X_num.skb.apply(SimpleImputer(strategy="median"))

X_proc = X_num_imp.skb.concat([X_obj_enc], axis=1)

# Model
model = LGBMRegressor(random_state=42)
pred_log = X_proc.skb.apply(model, y=y_log)

# Inverse log1p for predictions (only in predict mode)
mode = skrub.eval_mode()
function = mode.skb.match({"predict": np.expm1}, default=(lambda x: x))
pred = pred_log.skb.apply_func(function)

# CV
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
test = pd.read_csv("./input/test.csv")
test_ids = test["Id"]
test_X = test.drop("Id", axis=1)
test_preds = learner.predict({"_skrub_X": test_X})

submission = pd.DataFrame({"Id": test_ids, "SalePrice": test_preds})
submission.to_csv("./working/submission_skrub.csv", index=False)