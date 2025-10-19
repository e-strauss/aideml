import pandas as pd
import numpy as np
import skrub
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data as skrub var for lazy plan
data = skrub.var("data", pd.read_csv("./input/train.csv")).skb.subsample(n=100)
test_data = pd.read_csv("./input/test.csv")
test_ids = test_data["Id"]

# Target
y = data["SalePrice"].skb.apply_func(np.log1p).skb.mark_as_y()

# Feature matrix
X = data.drop(["Id", "SalePrice"], axis=1).skb.mark_as_X()

# Feature engineering steps, all as fine-grained plan ops
X_bsmt = X.assign(TotalBsmtSF=X["TotalBsmtSF"].skb.apply_func(lambda s: s.fillna(0)))
X_totalsf = X_bsmt.assign(TotalSF=X_bsmt["TotalBsmtSF"] + X_bsmt["1stFlrSF"] + X_bsmt["2ndFlrSF"])
X_garageyr = X_totalsf.assign(GarageYrBlt=X_totalsf["GarageYrBlt"].skb.apply_func(lambda s: s.fillna(X_totalsf["YrSold"])))
X_houseage = X_garageyr.assign(HouseAge=X_garageyr["YrSold"] - X_garageyr["YearBuilt"])
X_remodel = X_houseage.assign(SinceRemodel=X_houseage["YrSold"] - X_houseage["YearRemodAdd"])
X_garageage = X_remodel.assign(GarageAge=X_remodel["YrSold"] - X_remodel["GarageYrBlt"])
porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
X_porch = X_garageage.assign(TotalPorchSF=X_garageage[porch_cols].sum(axis=1))

# Save Neighborhood for target encoding, then drop from features
neigh = X_porch["Neighborhood"]
X_wo_neigh = X_porch.drop("Neighborhood", axis=1)

# Preprocessing: median impute numeric, label encode categoricals
obj_selector = skrub.selectors.filter(lambda col: col.dtype == "object")
num_selector = skrub.selectors.filter(lambda col: col.dtype != "object")

X_obj = X_wo_neigh.skb.select(obj_selector)
X_num = X_wo_neigh.skb.select(num_selector)

X_obj_imp = X_obj.skb.apply(SimpleImputer(strategy="constant", fill_value="Missing"))
X_obj_enc = X_obj_imp.skb.apply(OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
X_num_imp = X_num.skb.apply(SimpleImputer(strategy="median"))
X_pre = X_num_imp.skb.concat([X_obj_enc], axis=1)

# Target encoding for Neighborhood (fold-wise for CV, full for final)
class TargetEncoderFold:
    def __init__(self):
        self.mapping_ = None
        self.global_mean_ = None
    def fit(self, neigh, y):
        self.mapping_ = y.groupby(neigh).mean()
        self.global_mean_ = y.mean()
        return self
    def transform(self, neigh):
        return neigh.map(self.mapping_).fillna(self.global_mean_)

# Add target encoding as a plan op
def add_neigh_te(X, neigh, y, mode=None):
    # mode is injected by skrub.eval_mode()
    if mode is None or mode == "fit":
        # For fit, use full mapping
        mapping = y.groupby(neigh).mean()
        global_mean = y.mean()
        return X.assign(Neighborhood_TE=neigh.map(mapping).fillna(global_mean))
    elif mode == "predict":
        # For predict, mapping is from fit
        mapping = y.groupby(neigh).mean()
        global_mean = y.mean()
        return X.assign(Neighborhood_TE=neigh.map(mapping).fillna(global_mean))
    else:
        return X

# Add target encoding to the plan
mode = skrub.eval_mode()
X_te = X_pre.skb.apply_func(lambda X: add_neigh_te(X, neigh, y, mode=mode))

# Model
model = LGBMRegressor(random_state=42)
pred_log = X_te.skb.apply(model, y=y)

# Inverse log1p for predictions
function = mode.skb.match({"predict": np.expm1}, default=(lambda x: x))
pred = pred_log.skb.apply_func(function)

# CV
splits = pred.skb.get_data()
learner = pred.skb.make_learner()
data_ = pred.skb.get_data()
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = skrub.cross_validate(learner, data_, cv=cv, scoring=mean_squared_error, return_train_score=False)
rmse = np.sqrt(np.mean(scores["test_score"]))
print(f"CV RMSE: {rmse:.5f}")

# Final fit and test prediction
learner.fit(data_)
# Test feature engineering and preprocessing: do not repeat, just pass test data to pipeline
test_data_ = test_data.drop("Id", axis=1)
y_pred_test = learner.predict({"_skrub_X": test_data_})

submission = pd.DataFrame({"Id": test_ids, "SalePrice": y_pred_test})
submission.to_csv("./working/submission_skrub.csv", index=False)