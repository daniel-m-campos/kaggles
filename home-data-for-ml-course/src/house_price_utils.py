"""
Utility functions for the House Prices: Advanced Regression Techniques competition.
"""

import glob
from typing import Dict, Set

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import PredictionErrorDisplay, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def make_mi_scores(X, y, discrete_features):
    """
    Compute mutual information scores for a set of features and target.

    Parameters:
    X (DataFrame): The input features.
    y (Series): The target variable.
    discrete_features (array-like): A list of indices corresponding to discrete
        features.

    Returns:
    Series: A series containing the mutual information scores for each feature.
    """
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values()
    return mi_scores


def group_features(df: pd.DataFrame, ordinal_cutoff: int = 10) -> Dict[str, Set]:
    """
    Group the features of a DataFrame into nominal, ordinal, and continuous features.

    Parameters:
    df (DataFrame): The input DataFrame.
    ordinal_cutoff (int): The maximum number of unique values for a feature to be
        considered ordinal.

    Returns:
    dict: A dictionary containing the grouped features.
    """
    columns = set(df)
    cat_columns = set(df.select_dtypes(include=object))
    ordinal_cols = []
    cont_cols = []
    for column in columns - cat_columns:
        if df[column].nunique() < ordinal_cutoff:
            ordinal_cols.append(column)
        else:
            cont_cols.append(column)
    return {
        "nominal": sorted(cat_columns),
        "ordinal": sorted(ordinal_cols),
        "continuous": sorted(cont_cols),
    }


def clean(df: pd.DataFrame, features: Dict[str, Set]) -> pd.DataFrame:
    """
    Clean the input DataFrame by filling missing values and encoding categorical
        features.

    Parameters:
    df (DataFrame): The input DataFrame.
    features (dict): A dictionary containing the features of the DataFrame.

    Returns:
    DataFrame: The cleaned DataFrame.
    """

    num_columns = features["continuous"]
    df_cont = df[list(num_columns)].copy()
    df_cont = df_cont.fillna(df_cont.mean())
    assert all(df_cont.isnull().sum() == 0)

    # TODO: make this configurable since ordinal encdoing doesn't work well with linear
    # models
    encoder = OrdinalEncoder().set_output(transform="pandas")
    df_ord = encoder.fit_transform(df[list(features["ordinal"])])

    encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
    df_cat = encoder.fit_transform(df[list(features["nominal"])])

    df_clean = pd.concat([df_cont, df_ord, df_cat], axis=1)
    has_nans = [c for c in df_clean if df_clean[c].isnull().any()]
    return df_clean.drop(columns=has_nans)


def create_submission(model, X_test):
    """
    Create a submission file for a given model and test data.

    Parameters:
    model (object): The trained model used for prediction.
    X_test (DataFrame): The test data used for prediction.

    Returns:
    None
    """
    output = pd.DataFrame({"Id": X_test.index, "SalePrice": model.predict(X_test)})
    output.to_csv("submission.csv", index=False)


def residual_plots(y, yhat):
    """
    Generate residual plots for evaluating the performance of a regression model.

    Parameters:
    - y: Actual target values.
    - yhat: Predicted target values.

    Returns:
    - fig: Figure object containing the residual plots.
    """
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    PredictionErrorDisplay.from_predictions(
        y,
        y_pred=yhat,
        kind="actual_vs_predicted",
        ax=axs[0],
        random_state=0,
    )
    axs[0].set_title("Actual vs. Predicted values")
    PredictionErrorDisplay.from_predictions(
        y,
        y_pred=yhat,
        kind="residual_vs_predicted",
        ax=axs[1],
        random_state=0,
    )
    axs[1].set_title("Residuals vs. Predicted Values")
    fig.tight_layout()
    return fig


def load_data():
    """
    Load the train and test data from CSV files.

    Returns:
        tuple: A tuple containing the train and test data as pandas DataFrames.
    """
    train_file = glob.glob("../**/train.csv", recursive=True)[0]
    test_file = glob.glob("../**/test.csv", recursive=True)[0]
    return (
        pd.read_csv(train_file, index_col="Id"),
        pd.read_csv(test_file, index_col="Id"),
    )


def plot_variance(pca):
    """
    Plots the explained variance and cumulative variance of the principal components.

    Parameters:
    - pca: PCA object
        The PCA object containing the principal components.

    Returns:
    - fig: matplotlib Figure object
        The figure containing the variance plots.
    """

    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)

    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0))

    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0))

    fig.set(figwidth=8, dpi=100)
    return fig


def evaluate(model, X, y):
    return mean_absolute_error(y, model.predict(X))


def report(model, X_train, y_train, X_val, y_val):
    print(
        f"Train MAE = {evaluate(model, X_train, y_train):.2f}",
        f"Test MAE = {evaluate(model, X_val, y_val):.2f}",
        sep="\n",
    )


def setup_notebook():
    plt.style.use("ggplot")
    pd.options.display.max_columns = None
    pd.options.display.max_rows = 30


class QuantileKFold(StratifiedKFold):
    def __init__(self, n_splits=5, shuffle=False, n_quantiles=4, random_state=None):
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)
        self.n_quantiles = n_quantiles

    def split(self, X, y, groups=None):
        quantiles = pd.qcut(y, self.n_quantiles, labels=False)
        return super().split(X, quantiles, groups)
