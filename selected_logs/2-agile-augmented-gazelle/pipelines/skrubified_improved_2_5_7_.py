import skrub
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.preprocessing import TargetEncoder

# Load data
train = pd.read_csv("./input/train.csv", parse_dates=["datetime"])
test = pd.read_csv("./input/test.csv", parse_dates=["datetime"])

# Skrub DataOps plan
data = skrub.var("data", train).skb.subsample(n=1000)

y = data["count"].skb.mark_as_y()
y_log = y.skb.apply_func(np.log1p)
mode = skrub.eval_mode()

X = data.drop("count", axis=1).skb.mark_as_X()

# Pipeline 0
datetime_col = X["datetime"].dt
X_exploded_dt = X.assign(
    year=datetime_col.year,
    month=datetime_col.month,
    dayofweek=datetime_col.dayofweek,
    hour=datetime_col.hour)

X_feat_pipe0 = X_exploded_dt.drop(["datetime", "casual", "registered"], axis=1, errors="ignore")
hour = X_exploded_dt["hour"]
month = X_exploded_dt["month"]
dayofweek = X_exploded_dt["dayofweek"]

# Pipeline 2
model_pipe2 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
pred_pipe2 = X_feat_pipe0.skb.apply(model_pipe2, y=y_log).skb.set_name("Pipeline 2")
pred_final_pipe2 = pred_pipe2.skb.apply_func(
    lambda a,b: np.expm1(a) if b=="predict" else a,
    mode).skb.set_name("Reverse log for prediction 2")

# Pipeline 5
X_feat_pipe5 = X_feat_pipe0.assign(
    hour_sin= hour.apply(lambda x: np.sin(2 * np.pi * x / 24)),
    hour_cos = hour.apply(lambda x: np.cos(2 * np.pi * x / 24)),
    month_sin = month.apply(lambda x: np.sin(2 * np.pi * (x - 1) / 12)),
    month_cos = month.apply(lambda x: np.cos(2 * np.pi * (x - 1) / 12)),
)
model_pipe5 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
pred_pipe5 = X_feat_pipe5.skb.apply(model_pipe5, y=y_log).skb.set_name("Pipeline 5")
pred_final_pipe5 = pred_pipe5.skb.apply_func(
    lambda a,b: np.expm1(a) if b=="predict" else a,
    mode).skb.set_name("Reverse log for prediction 5")

# Pipeline 7
model_pipe7 = lgb.LGBMRegressor(learning_rate=0.05, num_leaves=31, n_estimators=1000, n_jobs=-1, random_state=42)
pred_pipe7 = X_feat_pipe5.skb.apply(model_pipe7,y=y_log).skb.set_name("Pipeline 7")

pred_final_pipe7 = pred_pipe7.skb.apply_func(
    lambda a,b: np.expm1(a) if b=="predict" else a,
    mode).skb.set_name("Reverse log for prediction 7")

# Pipeline 8
X_feat_pipe8 = X_feat_pipe5.assign(
    temp_diff=X["atemp"] - X["temp"],
)
pred_pipe8 = X_feat_pipe8.skb.apply(model_pipe7,y=y_log).skb.set_name("Pipeline 8")

pred_final_pipe8 = pred_pipe8.skb.apply_func(
    lambda a,b: np.expm1(a) if b=="predict" else a,
    mode).skb.set_name("Reverse log for prediction 8")

# Pipeline 9
te = TargetEncoder(target_type='continuous')
target_encoded_hour = X_feat_pipe5[["hour"]].skb.apply(te, y=y_log)
X_feat_pipe9 = X_feat_pipe5.skb.concat([target_encoded_hour], axis=1)
pred_pipe9 = X_feat_pipe9.skb.apply(model_pipe7,y=y_log).skb.set_name("Pipeline 9")

pred_final_pipe9 = pred_pipe9.skb.apply_func(
    lambda a,b: np.expm1(a) if b=="predict" else a,
    mode).skb.set_name("Reverse log for prediction 9")

# Pipeline 12
model_pipe12 = lgb.LGBMRegressor(
    objective="poisson",
    learning_rate=0.05,
    num_leaves=31,
    n_estimators=1000,
    n_jobs=-1,
    random_state=42,
    warnings="error",
    verbosity=-1
)
pred_pipe12 = X_feat_pipe5[["hour"]].skb.apply(model_pipe12, y=y).skb.set_name("Pipeline 12")


# Pipeline 13
X_feat_pipe13 = X_feat_pipe5.assign(
    dow_sin=dayofweek.apply(lambda x: np.sin(2 * np.pi * x / 7)),
    dow_cos=dayofweek.apply(lambda x: np.cos(2 * np.pi * x / 7)),
)
pred_pipe13 = X_feat_pipe13.skb.apply(model_pipe7,y=y_log).skb.set_name("Pipeline 13")

pred_final_pipe13 = pred_pipe13.skb.apply_func(
    lambda a,b: np.expm1(a) if b=="predict" else a,
    mode).skb.set_name("Reverse log for prediction 13")


merged_pipelines = skrub.choose_from({
    "pipeline2": pred_final_pipe2,
    "pipeline5": pred_final_pipe5,
    "pipeline7": pred_final_pipe7,
    "pipeline8": pred_final_pipe8,
    "pipeline9": pred_final_pipe9,
    "pipeline12": pred_pipe12,
    "pipeline13": pred_final_pipe13,
}, name="merged pipelines").as_data_op().skb.set_name("GridSearchCV")

print(merged_pipelines.skb.describe_param_grid())

graph = merged_pipelines.skb.draw_graph()
graph.open()

# RMSLE scorer
def rmsle(y_true, y_pred):
    print(f"y_true: {y_true.iloc[[0]].to_numpy()}, y_pred: {y_pred[0]}")
    return np.sqrt(mean_squared_log_error(y_true, np.clip(y_pred, 0, None)))
scorer = make_scorer(rmsle)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

search = merged_pipelines.skb.make_grid_search(fitted=True, cv=cv, scoring=scorer)
print(search.results_)
