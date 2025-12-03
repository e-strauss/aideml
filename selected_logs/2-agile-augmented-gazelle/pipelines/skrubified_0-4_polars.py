import skrub
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error, make_scorer
import time
import polars as pl
t0 = time.time()

# Load data
train = pl.read_csv(
    "./input/train.csv",
    schema_overrides={"datetime": pl.Datetime},
)

# Skrub DataOps plan
data = skrub.var("data", train).skb.subsample(n=1000)
y = data["count"].skb.mark_as_y()
y_log = y.skb.apply_func(np.log1p)
mode = skrub.eval_mode()
X = data.drop("count").skb.mark_as_X()

# Pipeline 0
X_feat_pipe0 = X.with_columns(
    year=pl.col("datetime").dt.year(),
    month=pl.col("datetime").dt.month(),
    dayofweek=pl.col("datetime").dt.weekday(),
    hour=pl.col("datetime").dt.hour())

X_feat_pipe0 = X_feat_pipe0.drop(["datetime", "casual", "registered"])
model_pipe0 = lgb.LGBMRegressor(random_state=42)
pred_pipe0 = X_feat_pipe0.skb.apply(model_pipe0, y=y).skb.set_name("Pipeline 0")

# Pipeline 1
model_pipe1 = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
pred_pipe1 = X_feat_pipe0.skb.apply(model_pipe1, y=y_log)
pred_final_pipe1 = pred_pipe1.skb.apply_func(
    lambda a,b: np.expm1(a) if b=="predict" else a,
    mode).skb.set_name("Reverse log for prediction1")

# Pipeline 2
model_pipe2 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
pred_pipe2 = X_feat_pipe0.skb.apply(model_pipe2, y=y_log).skb.set_name("Pipeline 2")
pred_final_pipe2 = pred_pipe2.skb.apply_func(
    lambda a,b: np.expm1(a) if b=="predict" else a,
    mode).skb.set_name("Reverse log for prediction2")

# Pipeline 3
model_pipe3 = Ridge()
pred_pipe3 = X_feat_pipe0.skb.apply(model_pipe3, y=y_log).skb.set_name("Pipeline 3")
pred_final_pipe3 = pred_pipe3.skb.apply_func(
    lambda a,b: np.expm1(a) if b=="predict" else a,
    mode).skb.set_name("Reverse log for prediction3")

# Pipeline 4
model_pipe4 = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
)
pred_pipe4 = X_feat_pipe0.skb.apply(model_pipe4, y=y_log).skb.set_name("Pipeline 4")
pred_final_pipe4 = pred_pipe4.skb.apply_func(
    lambda a,b: np.expm1(a) if b=="predict" else a,
    mode).skb.set_name("Reverse log for prediction4")

merged_pipelines = skrub.choose_from({
    "pipeline0": pred_pipe0,
    "pipeline1": pred_final_pipe1,
    "pipeline2": pred_final_pipe2,
    "pipeline3": pred_final_pipe3,
    "pipeline4": pred_final_pipe4,
}, name="merged pipelines").as_data_op().skb.set_name("GridSearchCV")

print(merged_pipelines.skb.describe_param_grid())

graph = merged_pipelines.skb.draw_graph()
graph.open()

# RMSLE scorer
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.clip(y_pred, 0, None)))
scorer = make_scorer(rmsle)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
t0_ = time.time()
search = merged_pipelines.skb.make_grid_search(fitted=True, cv=cv, scoring=scorer, n_jobs=-1)
print(search.results_)

t1 = time.time()
print(t1 - t0)
print(t1 - t0_)
