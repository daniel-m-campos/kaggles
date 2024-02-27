import glob
from typing import Dict, Set

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import PredictionErrorDisplay
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values()
    return mi_scores


def group_features(df: pd.DataFrame, ordinal_cutoff: int = 10) -> Dict[str, Set]:
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
        "nominal": cat_columns,
        "ordinal": set(ordinal_cols),
        "continuous": set(cont_cols),
    }


def clean(df: pd.DataFrame, features: Dict[str, Set]) -> pd.DataFrame:
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
    output = pd.DataFrame({"Id": X_test.index, "SalePrice": model.predict(X_test)})
    output.to_csv("submission.csv", index=False)


def residual_plots(y, yhat):
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
    fig.suptitle("Plotting cross-validated predictions")
    fig.tight_layout()
    return fig


def load_data():
    train_file = glob.glob("../**/train.csv", recursive=True)[0]
    test_file = glob.glob("../**/test.csv", recursive=True)[0]
    return (
        pd.read_csv(train_file, index_col="Id"),
        pd.read_csv(test_file, index_col="Id"),
    )
