import skrub
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
test_ids = test["Id"]

# Define skrub DataOps plan
data = skrub.var("data", train)
data = data.skb.subsample(n=100)

y = data["SalePrice"].skb.apply_func(np.log1p).skb.mark_as_y()
X = data.drop(["Id", "SalePrice"], axis=1).skb.mark_as_X()

# Preprocessing: handle categorical and numerical columns
cat_selector = skrub.selectors.filter(lambda col: col.dtype == "object")
num_selector = skrub.selectors.filter(lambda col: col.dtype != "object")

# Categorical: fillna("Missing"), convert to category, then codes
def fillna_missing_cat(col):
    return col.fillna("Missing").astype("category").cat.codes

X_cat = X.skb.select(cat_selector)
X_cat_enc = X_cat.skb.apply_func(fillna_missing_cat)

# Numerical: fillna with median
def fillna_median(col):
    med = col.median()
    return col.fillna(med)

X_num = X.skb.select(num_selector)
X_num_filled = X_num.skb.apply_func(fillna_median)

# Concatenate processed features
X_proc = X_cat_enc.skb.concat([X_num_filled], axis=1)

# Model
model = LGBMRegressor(random_state=42)
pred = X_proc.skb.apply(model, y=y)

# Cross-validation
learner = pred.skb.make_learner()
data_ = pred.skb.get_data()
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scorer = mean_squared_error

scores = skrub.cross_validate(learner, data_, cv=cv, scoring=scorer, return_train_score=False)
rmse = np.sqrt(np.mean(scores["test_score"]))
print(f"CV RMSE: {rmse:.5f}")

# Train on full data
learner.fit(data_)

# Predict on test set
y_pred_test = learner.predict({"_skrub_X": test})

# Output submission (apply expm1 to revert log1p)
submission = pd.DataFrame({"Id": test_ids, "SalePrice": np.expm1(y_pred_test)})
submission.to_csv("./working/submission_skrub.csv", index=False)