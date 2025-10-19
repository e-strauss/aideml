import skrub
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin

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

# Frequency encode Neighborhood
class FreqEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        col = X.columns[0]
        freq = X[col].value_counts(normalize=True)
        self.freq_ = freq
        return self
    def transform(self, X):
        col = X.columns[0]
        return pd.DataFrame({col+"_FE": X[col].map(self.freq_).fillna(0)})

X_neigh_fe = X[["Neighborhood"]].skb.apply(FreqEncoder())
X = X.skb.concat([X_neigh_fe], axis=1)

# Fill missing TotalBsmtSF as 0, then create TotalSF
X_totalbsmt = X.assign(TotalBsmtSF=X["TotalBsmtSF"].skb.apply_func(lambda s: s.fillna(0)))
X_totalsf = X_totalbsmt.assign(TotalSF=X_totalbsmt["TotalBsmtSF"] + X_totalbsmt["1stFlrSF"] + X_totalbsmt["2ndFlrSF"])

# GarageYrBlt fillna with YrSold
X_garageyr = X_totalsf.assign(GarageYrBlt=X_totalsf["GarageYrBlt"].skb.apply_func(
    lambda s: s.fillna(X_totalsf["YrSold"])
))

# Age-based features
X_houseage = X_garageyr.assign(HouseAge=X_garageyr["YrSold"] - X_garageyr["YearBuilt"])
X_remodel = X_houseage.assign(SinceRemodel=X_houseage["YrSold"] - X_houseage["YearRemodAdd"])
X_garageage = X_remodel.assign(GarageAge=X_remodel["YrSold"] - X_remodel["GarageYrBlt"])

# TotalPorchSF
porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
X_totalporch = X_garageage.assign(
    TotalPorchSF=X_garageage[porch_cols].skb.apply_func(lambda df: df.sum(axis=1))
)

# Preprocessing: median impute numeric, label encode categoricals
obj_selector = skrub.selectors.filter(lambda col: col.dtype == "object")
num_selector = skrub.selectors.filter(lambda col: col.dtype != "object")

X_obj = X_totalporch.skb.select(obj_selector)
X_num = X_totalporch.skb.select(num_selector)

X_obj_imp = X_obj.skb.apply(SimpleImputer(strategy="constant", fill_value="Missing"))
X_obj_enc = X_obj_imp.skb.apply(OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))

X_num_imp = X_num.skb.apply(SimpleImputer(strategy="median"))

X_final = X_num_imp.skb.concat([X_obj_enc], axis=1)

# Model
model = LGBMRegressor(random_state=42)
pred_log = X_final.skb.apply(model, y=y_log)

# Inverse log1p for predictions (only in predict mode)
mode = skrub.eval_mode()
inv_func = mode.skb.match({"predict": np.expm1}, default=(lambda x: x))
pred = pred_log.skb.apply_func(inv_func)

# CV
learner = pred.skb.make_learner()
data_ = pred.skb.get_data()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
def rmse_func(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
from sklearn.metrics import make_scorer
scorer = make_scorer(mean_squared_error, greater_is_better=False)
scores = skrub.cross_validate(learner, data_, cv=kf, scoring=scorer)
rmse = np.sqrt(-np.mean(scores["test_score"]))
print(f"CV RMSE: {rmse:.5f}")

# Train on full data
learner.fit(data_)

# Predict on test
test_data = pd.read_csv("./input/test.csv")
test_X = test_data.drop("Id", axis=1)
test_preds = learner.predict({"_skrub_X": test_X})

submission = pd.DataFrame({"Id": test_ids, "SalePrice": test_preds})
submission.to_csv("./working/submission_skrub.csv", index=False)