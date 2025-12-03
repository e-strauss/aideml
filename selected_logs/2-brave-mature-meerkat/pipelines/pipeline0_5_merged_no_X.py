import skrub
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
from time import time

t0 = time()
def amex_metric(y_true, y_pred):
    df = pd.DataFrame({"target": y_true, "prediction": y_pred})
    df = df.sort_values("prediction", ascending=False)
    df["weight"] = df["target"].apply(lambda x: 20 if x == 0 else 1)
    four_pct_cutoff = int(0.04 * df["weight"].sum())
    df["weight_cumsum"] = df["weight"].cumsum()
    df_cut = df[df["weight_cumsum"] <= four_pct_cutoff]
    top4 = df_cut["target"].sum() / df["target"].sum()
    df["random"] = (df["weight"] / df["weight"].sum()).cumsum()
    total_pos = (df["target"] * df["weight"]).sum()
    df["cum_pos_found"] = (df["target"] * df["weight"]).cumsum()
    df["lorentz"] = df["cum_pos_found"] / total_pos
    df["gini"] = (df["lorentz"] - df["random"]) * df["weight"]
    weighted_gini = df["gini"].sum()
    df_true = df.copy()
    df_true["prediction"] = df_true["target"]
    df_true = df_true.sort_values("prediction", ascending=False)
    df_true["weight"] = df_true["target"].apply(lambda x: 20 if x == 0 else 1)
    df_true["random"] = (df_true["weight"] / df_true["weight"].sum()).cumsum()
    total_pos2 = (df_true["target"] * df_true["weight"]).sum()
    df_true["cum_pos_found"] = (df_true["target"] * df_true["weight"]).cumsum()
    df_true["lorentz"] = df_true["cum_pos_found"] / total_pos2
    df_true["gini"] = (df_true["lorentz"] - df_true["random"]) * df_true["weight"]
    perfect_gini = df_true["gini"].sum()
    norm_gini = weighted_gini / perfect_gini
    return 0.5 * (norm_gini + top4)

# setup for multiple pipelines
data_path = skrub.as_data_op("input/train_data_downsampled.csv").skb.set_name("data")
data_path = data_path.skb.apply_func(lambda x: (x, print("data read"))[0])
labels_path = skrub.as_data_op("input/train_labels_downsampled.csv").skb.set_name("labels")

data = data_path.skb.apply_func(pd.read_csv)
labels = labels_path.skb.apply_func(pd.read_csv)

ids = data[["customer_ID"]].drop_duplicates().skb.subsample(n=1000)
joined = ids.merge(labels, on="customer_ID", how="inner")
ids = joined["customer_ID"].skb.mark_as_X()
y = joined["target"].skb.mark_as_y()

data = data[data["customer_ID"].isin(ids)]

# Skrub plan: pipeline 0
data_sorted = data.sort_values(["customer_ID", "S_2"])
data_latest = data_sorted.drop_duplicates("customer_ID", keep="last")

# align data_latest to ids
df = data_latest.set_index("customer_ID").loc[ids]

drop_cols = [
    "S_2",
    "B_30",
    "B_38",
    "D_114",
    "D_116",
    "D_117",
    "D_120",
    "D_126",
    "D_63",
    "D_64",
    "D_66",
    "D_68",
]

X = df.drop(columns=drop_cols)

model = lgb.LGBMClassifier(
    n_estimators=100, learning_rate=0.1, class_weight={0: 20, 1: 1}, random_state=42
)
pred0 = X.skb.apply(model, y=y)
# Skrub plan: pipeline 1
import xgboost as xgb

to_drop = [
    "B_30",
    "B_38",
    "D_114",
    "D_116",
    "D_117",
    "D_120",
    "D_126",
    "D_63",
    "D_64",
    "D_66",
    "D_68",
    "S_2",
]

df = data.drop(columns=to_drop, errors="ignore")
df_agg = df.groupby("customer_ID").mean().reset_index()

X_filled = df_agg.fillna(-999)

X_aligned = X_filled.set_index("customer_ID").loc[ids]

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=20,
    use_label_encoder=False,
    eval_metric="logloss",
    n_jobs=-1,
    random_state=42,
)
pred1 = X_aligned.skb.apply(model, y=y)


# Skrub plan: pipeline 3
cat_cols = [
    "S_2",
    "B_30",
    "B_38",
    "D_114",
    "D_116",
    "D_117",
    "D_120",
    "D_126",
    "D_63",
    "D_64",
    "D_66",
    "D_68",
]
data_nocat = data.drop(columns=cat_cols)

# Aggregate numerical features per customer
num = data_nocat.select_dtypes(include=[np.number])
agg = (num.groupby(data_nocat["customer_ID"])
       .agg(["mean", "std", "min", "max"])
       .pipe(lambda df: df.set_axis([f"{col}_{stat}" for col, stat in df.columns], axis=1))
       .reset_index())

# Impute missing
X_imp = agg.fillna(-999)
X_aligned = X_imp.set_index("customer_ID").loc[ids]

# Model
model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42, class_weight={0: 20, 1: 1})

# Skrub: pass sample_weight to model
pred3 = X_aligned.skb.apply(estimator=model, y=y)

# Skrub plan: pipeline 4
from catboost import CatBoostClassifier

cat_cols = [
    "S_2",
    "B_30",
    "B_38",
    "D_114",
    "D_116",
    "D_117",
    "D_120",
    "D_126",
    "D_63",
    "D_64",
    "D_66",
    "D_68",
]
feature_selector = skrub.selectors.all() -  (["customer_ID"] + cat_cols)
features = data.skb.apply_func(lambda df: feature_selector.expand(df))

# Sort and compute EWM features
df_sorted = data.sort_values(["customer_ID", "S_2"])
df_sorted = df_sorted.set_index("customer_ID")

df_ewm = df_sorted.groupby(level=0)[features].transform(lambda x: x.ewm(alpha=0.3).mean()).groupby(level=0).tail(1)
        

X_aligned = df_ewm.loc[ids]
# Model
model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=6,
    class_weights=[20, 1],
    verbose=0,
    random_seed=42,
)

pred4 = X_aligned.skb.apply(model, y=y)

pred0 = pred0.skb.apply_func(lambda x: (x, print("pipeline 0 done"))[0])
pred1 = pred1.skb.apply_func(lambda x: (x, print("pipeline 1 done"))[0])
pred3 = pred3.skb.apply_func(lambda x: (x, print("pipeline 3 done"))[0])
m = skrub.eval_mode()
pred4 = pred4.skb.apply_func(lambda x, mode: (x, print(f"pipeline 4 done [{mode}]"))[0], m)

preds = skrub.choose_from({
#    "pipeline0": pred0,
#    "pipeline1": pred1,
#    "pipeline3": pred3,
   "pipeline4": pred4,
}, name="merged pipelines").as_data_op().skb.set_name("GridSearchCV")

# Prepare data for cross-validation
data_ = preds.skb.get_data()

# Use the amex_metric as a scorer, needs_proba=True to pass proba to metric
scorer = make_scorer(amex_metric, needs_proba=True)
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
t1 = time()
print(f"Preview taken: {t1 - t0} seconds")
results = preds.skb.make_grid_search(fitted=True, cv=cv, scoring=scorer, n_jobs=-1, refit=False)
t2 = time()
print(f"Gridsearch taken: {t2 - t1} seconds")
print(f"Total time: {t2 - t0} seconds")
print(results.results_)