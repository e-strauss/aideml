import pandas as pd
import numpy as np
import skrub
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
from lightgbm import LGBMRegressor

# Custom target encoder for a single categorical column
class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mapping_ = None
        self.global_mean_ = None
        self.col_ = None

    def fit(self, X: pd.DataFrame, y):
        col = X.columns[0]
        self.col_ = col
        df = pd.DataFrame({"cat": X[col], "y": y})
        self.mapping_ = df.groupby("cat")["y"].mean()
        self.global_mean_ = df["y"].mean()
        return self

    def transform(self, X: pd.DataFrame):
        col = X.columns[0]
        vals = X[col].map(self.mapping_).fillna(self.global_mean_)
        return pd.DataFrame({f"{col}_TE": vals})

# Load training data
data = pd.read_csv("./input/train.csv")

# Create DataOps variable and subsample for fast preview
data_var = skrub.var("data", data).skb.subsample(n=200)

# Target and log-transform
y = data_var["SalePrice"].skb.mark_as_y()
y_log = y.skb.apply_func(np.log1p)

# Base features
X = data_var.drop(["Id", "SalePrice"], axis=1).skb.mark_as_X()

# Feature engineering
X = X.assign(TotalBsmtSF=X["TotalBsmtSF"].fillna(0))
X = X.assign(TotalSF=X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"])
X = X.assign(GarageYrBlt=X["GarageYrBlt"].fillna(X["YrSold"]))
X = X.assign(HouseAge=X["YrSold"] - X["YearBuilt"])
X = X.assign(SinceRemodel=X["YrSold"] - X["YearRemodAdd"])
X = X.assign(GarageAge=X["YrSold"] - X["GarageYrBlt"])
porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
X = X.assign(TotalPorchSF=X[porch_cols].sum(axis=1))

# Target encoding for Neighborhood (single column)
neigh_selector = skrub.selectors.filter_names(lambda name: name == "Neighborhood")
X_neigh = X.skb.select(neigh_selector)
neigh_te = X_neigh.skb.apply(TargetMeanEncoder(), y=y_log)

# Drop Neighborhood from main features
X_wo_neigh = X.drop("Neighborhood", axis=1)

# Preprocessing: median impute numeric, ordinal encode other categoricals
num_selector = skrub.selectors.filter(lambda col: col.dtype != "object")
obj_selector = skrub.selectors.filter(lambda col: col.dtype == "object")

X_num = X_wo_neigh.skb.select(num_selector)
X_num_imp = X_num.skb.apply(SimpleImputer(strategy="median"))

X_obj = X_wo_neigh.skb.select(obj_selector)
X_obj_imp = X_obj.skb.apply(SimpleImputer(strategy="constant", fill_value="Missing"))
X_obj_enc = X_obj_imp.skb.apply(OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))

# Concatenate all features
X_vec = X_num_imp.skb.concat([X_obj_enc, neigh_te], axis=1)

# Model
model = LGBMRegressor(random_state=42)
pred_log = X_vec.skb.apply(model, y=y_log)

# Build learner and evaluate with 5-fold CV (RMSE on log1p target)
learner = pred_log.skb.make_learner()
data_ = pred_log.skb.get_data()

scorer = make_scorer(mean_squared_error)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = skrub.cross_validate(learner, data_, cv=cv, scoring=scorer, return_train_score=False)
rmse = np.sqrt(np.mean(scores["test_score"]))
print(f"CV RMSE: {rmse:.5f}")

# Fit final model on full data
learner.fit(data_)

# Predict on test and invert the log1p
test_data = pd.read_csv("./input/test.csv")
test_ids = test_data["Id"]
test_features = test_data.drop("Id", axis=1)
test_pred_log = learner.predict({"_skrub_X": test_features})
test_preds = np.expm1(test_pred_log)

# Save submission
submission = pd.DataFrame({"Id": test_ids, "SalePrice": test_preds})
submission.to_csv("./working/submission.csv", index=False)