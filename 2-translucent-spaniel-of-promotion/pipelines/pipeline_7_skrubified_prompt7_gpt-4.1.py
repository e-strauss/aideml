import pandas as pd
import numpy as np
import skrub
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
test_ids = test["Id"]

# Define skrub variable for train data, subsample for preview
data = skrub.var("data", train).skb.subsample(n=100)

# Target: log1p transform
y = data["SalePrice"].skb.apply_func(np.log1p).skb.mark_as_y()

# Features: drop Id and SalePrice, mark as X
X = data.drop(["Id", "SalePrice"], axis=1).skb.mark_as_X()

# Feature engineering: fillna TotalBsmtSF as 0, then create TotalSF
X_bsmt = X.assign(TotalBsmtSF=X["TotalBsmtSF"].skb.apply_func(lambda s: s.fillna(0)))
X_totalsf = X_bsmt.assign(TotalSF=X_bsmt["TotalBsmtSF"] + X_bsmt["1stFlrSF"] + X_bsmt["2ndFlrSF"])

# Preprocessing: median impute numeric, ordinal encode categoricals
obj_selector = skrub.selectors.filter(lambda col: col.dtype == "object")
num_selector = skrub.selectors.filter(lambda col: col.dtype != "object")

X_obj = X_totalsf.skb.select(obj_selector)
X_num = X_totalsf.skb.select(num_selector)

X_obj_imp = X_obj.skb.apply(SimpleImputer(strategy="constant", fill_value="Missing"))
X_obj_enc = X_obj_imp.skb.apply(OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))

X_num_imp = X_num.skb.apply(SimpleImputer(strategy="median"))

X_proc = X_num_imp.skb.concat([X_obj_enc], axis=1)

# Model
model = LGBMRegressor(random_state=42)
pred_log = X_proc.skb.apply(model, y=y)

# Inverse log1p for predictions (only in predict mode)
mode = skrub.eval_mode()
inv_func = mode.skb.match({"predict": np.expm1}, default=(lambda x: x))
pred = pred_log.skb.apply_func(inv_func)

# Cross-validation
learner = pred.skb.make_learner()
data_ = pred.skb.get_data()
scorer = make_scorer(mean_squared_error)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = skrub.cross_validate(learner, data_, cv=cv, scoring=scorer)
rmse = np.sqrt(np.mean(scores["test_score"]))
print(f"CV RMSE: {rmse:.5f}")

# Train on full data
learner.fit(data_)

# Predict on test
test_data = test.copy()
y_pred_test = learner.predict({"_skrub_X": test_data})

submission = pd.DataFrame({"Id": test_ids, "SalePrice": y_pred_test})
submission.to_csv("./working/submission_skrub.csv", index=False)