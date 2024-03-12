# %%
import catboost as cb
import house_price_utils as hpu
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.compose import TransformedTargetRegressor, make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

from ml_tools import column_selectors
from ml_tools.experiment import RegressionExperiment

# %%
data, data_test = hpu.load_data()
X, y = hpu.extract_target(data, "SalePrice")


# %%
preprocessor = make_column_transformer(
    (
        make_pipeline(SimpleImputer(strategy="mean")),
        column_selectors.continuous_selector,
    ),
    (
        make_pipeline(
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ),
        column_selectors.nominal_selector,
    ),
    (
        make_pipeline(
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ),
        column_selectors.ordinal_selector,
    ),
)

# %% Default tree models
tree_models = [
    xgb.XGBRegressor(),
    HistGradientBoostingRegressor(),
    RandomForestRegressor(),
]

# %%
for model in tree_models:
    RegressionExperiment(
        f"{model.__class__.__name__} plus target log transform",
        preprocessor,
        TransformedTargetRegressor(regressor=model, func=np.log, inverse_func=np.exp),
        mean_absolute_error,
        num_y_quantiles=10,
    ).cross_validate(X, y).save(root_dir="../models")

# %% Tune XGBoost

RegressionExperiment(
    "XGBRegressor Manual Tuning",
    preprocessor,
    TransformedTargetRegressor(
        regressor=xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            n_jobs=-1,
            objective="reg:absoluteerror",
            max_depth=3,
            importance_type="weight",
        ),
        func=np.log,
        inverse_func=np.exp,
    ),
    mean_absolute_error,
    num_y_quantiles=5,
).cross_validate(X, y).save(root_dir="../models")

# %% Target encoding
preprocessor2 = make_column_transformer(
    (TargetEncoder(target_type="continuous"), ["Neighborhood"]),
    (
        make_pipeline(SimpleImputer(strategy="mean")),
        column_selectors.continuous_selector,
    ),
    (
        make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ),
        column_selectors.nominal_selector,
    ),
    (
        make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ),
        column_selectors.ordinal_selector,
    ),
)

# %%
RegressionExperiment(
    "XGBRegressor with Target encoding",
    preprocessor2,
    xgb.XGBRegressor(
        n_jobs=-1,
        objective="reg:absoluteerror",
        importance_type="weight",
    ),
    mean_absolute_error,
    num_y_quantiles=5,
).cross_validate(X, y).save(root_dir="../models")

# %%

RegressionExperiment(
    "LGBMRegressor",
    preprocessor,
    TransformedTargetRegressor(
        regressor=lgb.LGBMRegressor(
            objective="regression",
            importance_type="gain",
            n_estimators=300,
            learning_rate=0.05,
        ),
        func=np.log,
        inverse_func=np.exp,
    ),
    mean_absolute_error,
    num_y_quantiles=5,
).cross_validate(X, y).save(root_dir="../models")

# %%
RegressionExperiment(
    "CatBoostRegressor",
    preprocessor,
    TransformedTargetRegressor(
        regressor=cb.CatBoostRegressor(
            loss_function="RMSE",
            n_estimators=300,
            learning_rate=0.05,
        ),
        func=np.log,
        inverse_func=np.exp,
    ),
    mean_absolute_error,
    num_y_quantiles=5,
).cross_validate(X, y).save(root_dir="../models")

# %%
RegressionExperiment(
    "CatBoostRegressor with tuned hyperparameters",
    preprocessor,
    TransformedTargetRegressor(
        regressor=cb.CatBoostRegressor(
            loss_function="RMSE",
            **{
                "n_estimators": 966,
                "depth": 6,
                "learning_rate": 0.030582247474575365,
                "l2_leaf_reg": 0.7010825540551232,
            },
        ),
        func=np.log,
        inverse_func=np.exp,
    ),
    mean_absolute_error,
    num_y_quantiles=5,
).cross_validate(X, y).save(root_dir="../models")

# %%
