import skrub
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, make_scorer
from xgboost import XGBRegressor

# Load data and create skrub variable for lazy plan
data = pd.read_csv(
    "./input/price_paid_records.csv", parse_dates=["Date of Transfer"], low_memory=False
)
data_var = skrub.var("data", data).skb.subsample(n=1000)

# Feature engineering
X = data_var.assign(
    year=data_var["Date of Transfer"].dt.year,
    month=data_var["Date of Transfer"].dt.month,
    day=data_var["Date of Transfer"].dt.day,
)
X = X.drop(
    [
        "Date of Transfer",
        "Transaction unique identifier",
        "Record Status - monthly file only",
    ],
    axis=1,
).skb.mark_as_X()

y = data_var["Price"].skb.mark_as_y()

# Label encode categorical features using skrub selectors and apply
cat_selector = skrub.selectors.filter(lambda col: col.dtype == "object")
X_cat = X.skb.select(cat_selector)
X_cat_enc = X_cat.skb.apply(LabelEncoder())

num_selector = skrub.selectors.filter(lambda col: col.dtype != "object")
X_num = X.skb.select(num_selector)

X_vec = X_num.skb.concat([X_cat_enc], axis=1)

# Model
model = XGBRegressor(
    n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0
)
pred = X_vec.skb.apply(model, y=y)

# Make learner
learner = pred.skb.make_learner()
data_ = pred.skb.get_data()

# Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scorer = make_scorer(mean_squared_error, squared=False)
scores = skrub.cross_validate(learner, data_, cv=cv, scoring=scorer, return_train_score=False)

for i, rmse in enumerate(scores["test_score"]):
    print(f"Fold RMSE: {rmse:.4f}")

print(f"Average CV RMSE: {np.mean(scores['test_score']):.4f}")