import skrub
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
test_ids = test["Id"]

# Define skrub DataOps plan
data = skrub.var("data", train).skb.subsample(n=100)

# Target
y = data["SalePrice"].skb.apply_func(np.log1p).skb.mark_as_y()

# Features (drop Id and SalePrice)
X = data.drop(["Id", "SalePrice"], axis=1).skb.mark_as_X()

# Numeric/categorical selectors
numeric_selector = skrub.selectors.filter(lambda col: np.issubdtype(col.dtype, np.number))
categorical_selector = skrub.selectors.filter(lambda col: col.dtype == "object")

# Numeric imputation
X_num = X.skb.select(numeric_selector)
X_num_imputed = X_num.skb.apply_func(lambda df: df.fillna(df.median()))

# Categorical imputation + one-hot
X_cat = X.skb.select(categorical_selector)
X_cat_filled = X_cat.skb.apply_func(lambda df: df.fillna("Missing"))
X_cat_ohe = X_cat_filled.skb.apply_func(lambda df: pd.get_dummies(df, drop_first=False))

# Concatenate numeric and categorical
X_vec = X_num_imputed.skb.concat([X_cat_ohe], axis=1)

# Align columns for train/test
# Skrub will handle this automatically if we use the same pipeline for test

# Model
model = Ridge(alpha=1.0, random_state=42)
pred_log = X_vec.skb.apply(model, y=y)

# Inverse log1p for predictions (only in predict mode)
mode = skrub.eval_mode()
inv_func = mode.skb.match({"predict": np.expm1}, default=(lambda x: x))
pred = pred_log.skb.apply_func(inv_func)

# CV
splits = pred.skb.get_data()
learner = pred.skb.make_learner()
cv = KFold(n_splits=5, shuffle=True, random_state=42)
def scorer(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)
scores = skrub.cross_validate(learner, splits, cv=cv, scoring=scorer)
print(f"CV RMSE (log scale): {np.mean(scores['test_score']):.5f}")

# Retrain on full data
learner.fit(splits)

# Predict on test
test_data = test.copy()
test_data = test_data.drop(["Id"], axis=1)
test_preds = learner.predict({"_skrub_X": test_data})

# Save submission
submission = pd.DataFrame({"Id": test_ids, "SalePrice": test_preds})
submission.to_csv("./working/submission_skrub.csv", index=False)