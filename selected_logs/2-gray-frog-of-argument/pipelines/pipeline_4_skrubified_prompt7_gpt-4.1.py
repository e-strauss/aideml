import skrub
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer

# Load data and create skrub variable (subsample for preview)
data = pd.read_csv("./input/price_paid_records.csv", nrows=500000)
data_var = skrub.var("data", data).skb.subsample(n=10000)

# Parse dates
data_var = data_var.assign(
    **{
        "Date of Transfer": pd.to_datetime(data_var["Date of Transfer"]),
        "year": data_var["Date of Transfer"].dt.year,
        "month": data_var["Date of Transfer"].dt.month,
        "day": data_var["Date of Transfer"].dt.day,
    }
)

# Drop unused columns
X = data_var.drop(
    [
        "Transaction unique identifier",
        "Record Status - monthly file only",
        "Date of Transfer",
        "Price",
    ],
    axis=1,
).skb.mark_as_X()
y = data_var["Price"].skb.mark_as_y()

# Label encode categoricals
obj_selector = skrub.selectors.filter(lambda col: col.dtype == "object")
X_obj = X.skb.select(obj_selector)
X_obj_enc = X_obj.skb.apply(LabelEncoder())
num_selector = skrub.selectors.filter(lambda col: col.dtype != "object")
X_num = X.skb.select(num_selector)
X_vec = X_num.skb.concat([X_obj_enc], axis=1)

# Model
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
pred = X_vec.skb.apply(model, y=y)

# Make learner
learner = pred.skb.make_learner()
data_ = pred.skb.get_data()

# CV with RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

scorer = make_scorer(rmse, greater_is_better=False)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = skrub.cross_validate(learner, data_, cv=cv, scoring=scorer, return_train_score=False)
print(f"Average CV RMSE: {np.abs(np.mean(scores['test_score'])):.2f}")