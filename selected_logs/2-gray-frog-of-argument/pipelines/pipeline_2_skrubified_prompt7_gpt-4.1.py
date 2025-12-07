import skrub
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, make_scorer

DATA_PATH = "./input/price_paid_records.csv"
NROWS = 2_000_000
data = pd.read_csv(DATA_PATH, parse_dates=["Date of Transfer"], nrows=NROWS)
data = skrub.var("data", data).skb.subsample(n=10000)  # subsample for preview

# Feature engineering
data_year = data.assign(year=data["Date of Transfer"].dt.year)
data_month = data_year.assign(month=data_year["Date of Transfer"].dt.month)
data_feat = data_month.drop(
    columns=[
        "Transaction unique identifier",
        "Record Status - monthly file only",
        "Date of Transfer",
    ]
)

y = data_feat["Price"].skb.mark_as_y()
X = data_feat.drop("Price", axis=1).skb.mark_as_X()

# Label encode categorical features
cat_selector = skrub.selectors.filter(lambda col: col.dtype == "object")
X_cat = X.skb.select(cat_selector)
# LabelEncoder is not natively multi-column, so wrap it
from sklearn.base import BaseEstimator, TransformerMixin
class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.encoders_ = {}
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders_[col] = le
        return self
    def transform(self, X):
        X_out = X.copy()
        for col in X.columns:
            X_out[col] = self.encoders_[col].transform(X_out[col].astype(str))
        return X_out

X_cat_enc = X_cat.skb.apply(MultiLabelEncoder())
num_selector = skrub.selectors.filter(lambda col: col.dtype != "object")
X_num = X.skb.select(num_selector)
X_vec = X_num.skb.concat([X_cat_enc], axis=1)

model = HistGradientBoostingRegressor(random_state=42)
pred = X_vec.skb.apply(model, y=y)

learner = pred.skb.make_learner()
data_ = pred.skb.get_data()

def rmse_func(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
scorer = make_scorer(rmse_func, greater_is_better=False)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = skrub.cross_validate(learner, data_, cv=cv, scoring=scorer, return_train_score=False)
rmses = -scores["test_score"]
for i, rmse in enumerate(rmses):
    print(f"Fold RMSE: {rmse:.2f}")
print(f"Average CV RMSE: {np.mean(rmses):.2f}")