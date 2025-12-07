import skrub
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, make_scorer

usecols = [
    "Date of Transfer",
    "County",
    "District",
    "Duration",
    "Old/New",
    "PPDCategory Type",
    "Property Type",
    "Town/City",
    "Price",
]

data_path = "input/price_paid_records.csv"
data = skrub.as_data_op(data_path).skb.apply_func(
    pd.read_csv, usecols=usecols,
    parse_dates=["Date of Transfer"],
    low_memory=False
).skb.subsample(n=10000)

# Feature engineering
X = data.assign(
    year=data["Date of Transfer"].dt.year,
    month=data["Date of Transfer"].dt.month,
).drop(["Date of Transfer", "Price"], axis=1).skb.mark_as_X()
y = data["Price"].skb.mark_as_y()

cat_cols = [
    "County",
    "District",
    "Duration",
    "Old/New",
    "PPDCategory Type",
    "Property Type",
    "Town/City",
]

# Convert to categorical
for col in cat_cols:
    X = X.assign(**{col: X[col].astype("category")})

model = lgb.LGBMRegressor(objective="regression", n_estimators=100, n_jobs=-1)

# Pass categorical_feature to LGBMRegressor via fit_params
pred = X.skb.apply(model, y=y)

print("making learner")
learner = pred.skb.make_learner()
print("getting data")
data_ = pred.skb.get_data()

scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
print("cross-validating")
scores = skrub.cross_validate(
    learner,
    data_,
    cv=cv,
    scoring=scorer,
    return_train_score=False,
)

print(f"CV RMSE: {np.mean(-scores['test_score']):.4f}")