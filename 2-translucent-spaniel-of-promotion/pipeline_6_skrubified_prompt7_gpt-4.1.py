import pandas as pd
import numpy as np
import skrub
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, make_scorer

# Load data
train_df = pd.read_csv("input/train.csv")
test_df = pd.read_csv("input/test.csv")
test_ids = test_df["Id"]

# Start skrub plan
data = skrub.var("data", train_df).skb.subsample(n=100)

# Identify feature types
cat_selector = skrub.selectors.filter(lambda col: col.dtype == "object")
num_selector = skrub.selectors.filter(lambda col: col.dtype != "object" and col.name not in ["Id", "SalePrice"])

# Prepare y and X
y = data["SalePrice"].skb.apply_func(np.log1p).skb.mark_as_y()
X = data.drop(["Id", "SalePrice"], axis=1).skb.mark_as_X()

# Numeric imputation
X_num = X.skb.select(num_selector)
num_imputer = SimpleImputer(strategy="median")
X_num_imp = X_num.skb.apply(num_imputer)

# Categorical imputation and encoding
X_cat = X.skb.select(cat_selector)
cat_imputer = SimpleImputer(strategy="constant", fill_value="Missing")
X_cat_imp = X_cat.skb.apply(cat_imputer)
cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_cat_enc = X_cat_imp.skb.apply(cat_encoder)

# Concatenate numeric and categorical
X_all = X_num_imp.skb.concat([X_cat_enc], axis=1)

# Log-transform highly skewed numeric features
skewness = X_num_imp.skew()
skewed = skewness.abs() > 0.75
skewed_cols = skewness[skewed].index.tolist()
not_skewed_cols = skewness[~skewed].index.tolist()
X_skewed = X_num_imp.skb.select(skewed_cols)
X_skewed_log = X_skewed.apply(np.log1p)
X_not_skewed = X_num_imp.skb.select(not_skewed_cols)
X_num_final = X_not_skewed.skb.concat([X_skewed_log], axis=1)
X_all_final = X_num_final.skb.concat([X_cat_enc], axis=1)

# Model
model = LGBMRegressor(random_state=42)
pred_log = X_all_final.skb.apply(model, y=y)

# Inverse log1p for predictions (only in predict mode)
mode = skrub.eval_mode()
inv_func = mode.skb.match({"predict": np.expm1}, default=(lambda x: x))
pred = pred_log.skb.apply_func(inv_func)

# CV
splits = pred.skb.get_data()
learner = pred.skb.make_learner()
scorer = make_scorer(mean_squared_error)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = skrub.cross_validate(learner, splits, cv=cv, scoring=scorer, return_train_score=True)
rmse = np.sqrt(np.mean(scores["test_score"]))
print(f"CV RMSE: {rmse:.5f}")

# Train on full data
learner.fit(splits)

# Predict on test
test_data = test_df.drop("Id", axis=1)
test_preds = learner.predict({"_skrub_X": test_data})

submission = pd.DataFrame({"Id": test_ids, "SalePrice": test_preds})
submission.to_csv("./working/submission_skrub.csv", index=False)