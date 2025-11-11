import skrub
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error, make_scorer

# Load data
train = pd.read_csv("./input/train.csv", parse_dates=["datetime"])
test = pd.read_csv("./input/test.csv", parse_dates=["datetime"])

# Skrub DataOps plan
data = skrub.var("data", train).skb.subsample(n=100)

# Feature engineering (fine-grained, no UDFs)
X = data.skb.mark_as_X()
X_year = X.assign(year=X["datetime"].dt.year)
X_month = X_year.assign(month=X_year["datetime"].dt.month)
X_dayofweek = X_month.assign(dayofweek=X_month["datetime"].dt.dayofweek)
X_hour = X_dayofweek.assign(hour=X_dayofweek["datetime"].dt.hour)

features = [
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "year",
    "month",
    "dayofweek",
    "hour",
]
X_feat = X_hour[features]

y = data["count"].skb.mark_as_y()

model = lgb.LGBMRegressor(random_state=42)
pred = X_feat.skb.apply(model, y=y)

learner = pred.skb.make_learner()
data_ = pred.skb.get_data()

# RMSLE scorer
def rmsle(y_true, y_pred):
    error = mean_squared_log_error(y_true, np.clip(y_pred, 0, None))
    sqrt_error = np.sqrt(error)
    return sqrt_error
scorer = make_scorer(rmsle)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = skrub.cross_validate(learner, data_, cv=cv, scoring=scorer)
print(scores['test_score'])
print(f"5-fold RMSLE: {np.mean(scores['test_score']):.5f}")

# Train on full data
learner.fit(data_)

# Do not create a skrub.var from test data, just pass as dict to learner.predict
test_preds = learner.predict({"_skrub_X": test})
test_preds = np.clip(test_preds, 0, None).round().astype(int)

submission = pd.DataFrame(
    {"datetime": test["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"), "count": test_preds}
)
submission.to_csv("./working/submission_skrub.csv", index=False)