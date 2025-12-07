import skrub
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import mean_squared_error, make_scorer

# Load data and create skrub variable for lazy plan
data = pd.read_csv("./input/price_paid_records.csv", nrows=2_000_000)
data = skrub.var("data", data).skb.subsample(n=100_000)  # subsample for preview

# Feature engineering: parse date
date_col = data["Date of Transfer"].skb.apply_func(pd.to_datetime)
X_with_date = data.assign(
    year=date_col.dt.year,
    month=date_col.dt.month,
    day=date_col.dt.day,
)

# Drop unused or constant columns
X_drop = X_with_date.drop(
    [
        "Date of Transfer",
        "Transaction unique identifier",
        "Record Status - monthly file only",
    ],
    axis=1,
)

# Separate target and features
y = X_drop["Price"].skb.mark_as_y()
X = X_drop.drop("Price", axis=1).skb.mark_as_X()

# Label encode categorical features using OrdinalEncoder (skrub will broadcast to all object columns)
obj_selector = skrub.selectors.filter(lambda col: col.dtype == "object")
X_obj = X.skb.select(obj_selector)
X_obj_enc = X_obj.skb.apply(OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))

num_selector = skrub.selectors.filter(lambda col: col.dtype != "object")
X_num = X.skb.select(num_selector)

X_vec = X_num.skb.concat([X_obj_enc], axis=1)

# Model
model = Ridge()
pred = X_vec.skb.apply(model, y=y)

# Make learner
learner = pred.skb.make_learner()
data_ = pred.skb.get_data()

# Cross-validation
scorer = make_scorer(mean_squared_error, greater_is_better=False)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = skrub.cross_validate(
    learner,
    data_,
    cv=cv,
    scoring=lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    return_train_score=False,
)

print(f"CV RMSE: {np.mean(scores['test_score']):.2f}")