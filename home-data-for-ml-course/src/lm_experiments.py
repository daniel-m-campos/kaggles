#  %%
import house_price_utils as hpu
import numpy as np
from sklearn import linear_model, metrics
from sklearn.compose import TransformedTargetRegressor, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from ml_tools import column_selectors
from ml_tools.experiment import RegressionExperiment

# %%
data, data_test = hpu.load_data()
X, y = hpu.extract_target(data, "SalePrice")

# %%
preprocessor1 = make_column_transformer(
    (
        make_pipeline(SimpleImputer(strategy="mean"), StandardScaler()),
        column_selectors.continuous_selector,
    ),
    (
        make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore"),
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
linear_models = [
    linear_model.LinearRegression(),
    linear_model.ElasticNetCV(),
]

# %%
for model in linear_models:
    RegressionExperiment(
        model.__class__.__name__,
        KFold(n_splits=5),
        preprocessor1,
        TransformedTargetRegressor(regressor=model, func=np.log, inverse_func=np.exp),
        metrics.mean_absolute_error,
    ).fit(X, y).save(root_dir="../models")
