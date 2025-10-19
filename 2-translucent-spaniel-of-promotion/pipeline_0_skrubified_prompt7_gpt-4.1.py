import pandas as pd
import numpy as np
import skrub
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, make_scorer

# Load data
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")
test_ids = test["Id"]

# Start DataOps plan
data = skrub.var("data", train).skb.subsample(n=100)

# Target
y = data["SalePrice"].skb.mark_as_y()
y_log = y.skb.apply_func(np.log1p)

# Features
X = data.drop(["Id", "SalePrice"], axis=1).skb.mark_as_X()

# Preprocessing: categorical columns
cat_selector = skrub.selectors.filter(lambda col: col.dtype == "object")
X_cat = X.skb.select(cat_selector)
# fillna("Missing") and label encode
X_cat_filled = X_cat.skb.apply_func(lambda df: df.fillna("Missing"))
X_cat_encoded = X_cat_filled.skb.apply_func(lambda df: df.apply(lambda col: col.astype("category").cat.codes))

# Preprocessing: numerical columns
num_selector = skrub.selectors.filter(lambda col: col.dtype != "object")
X_num = X.skb.select(num_selector)
# fillna with median
X_num_filled = X_num.skb.apply_func(lambda df: df.fillna(df.median()))

# Concatenate processed features
X_proc = X_cat_encoded.skb.concat([X_num_filled], axis=1)

# Model
model = LGBMRegressor(random_state=42)
pred_log = X_proc.skb.apply(model, y=y_log)
# Inverse log1p for predictions (only in predict mode)
mode = skrub.eval_mode()
function = mode.skb.match({"predict": np.expm1}, default=(lambda x: x))
pred = pred_log.skb.apply_func(function)

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
test_data = test.drop("Id", axis=1)
test_preds = learner.predict({"_skrub_X": test_data})

submission = pd.DataFrame({"Id": test_ids, "SalePrice": test_preds})
submission.to_csv("./working/submission_skrub.csv", index=False)