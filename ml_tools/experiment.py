import glob
from pathlib import Path
from typing import Callable, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold


class RegressionExperiment:
    """
    A class representing a machine learning experiment for regression.

    Why is this needed? Basically, to minimize the amount of code that needs to be
    written for iterating over different models and preprocessing steps and ensure
    reproducibility. This class encapsulates the entire process of fitting a model to
    the training data and generating predictions. It also allows the experiment to be
    saved to and loaded from a file, which is useful for sharing and reproducing
    results.

    In particular, random_state is default set to not None, and the random_state
    attribute is set for both the cross-validation strategy and the regression
    estimator to avoid the issues describe in the following link:
    https://scikit-learn.org/1.4/common_pitfalls.html#general-recommendations


    Attributes:
        name (str): The name of the experiment.
        preprocessor: The data preprocessor.
        estimator: The regression estimator.
        score_func: The scoring function.
        cv: The cross-validation strategy.
        scores: The list of scores obtained during cross-validation.
        models: The list of trained models obtained during cross-validation.
        oof_predictions: The out-of-fold predictions obtained during cross-validation.
        residuals: The residuals obtained during cross-validation.
        random_state: The random state.

    Methods:
        fit: Fits the experiment on the training data.
        predict: Generates mean ensemble predictions.
        save: Saves the experiment to a file.
        load: Loads a saved experiment from a file.
    """

    def __init__(
        self,
        name: str,
        preprocessor: Union[BaseEstimator, TransformerMixin],
        estimator: BaseEstimator,
        score_func: Callable[[np.array, np.array], float],
        cv: Optional[BaseCrossValidator] = None,
        num_y_quantiles: Optional[int] = None,
        random_state: int = 42,
    ):
        self.name = name
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.score_func = score_func
        self.cv = cv or StratifiedKFold(n_splits=5, shuffle=True)
        self.num_y_quantiles = num_y_quantiles
        self.scores = None
        self.models = None
        self.oof_predictions = None
        self.residuals = None
        self.random_state = random_state

        for obj in (self.cv, self.preprocessor, self.estimator):
            self._set_random_state(obj)

    def _set_random_state(self, obj):
        """
        Sets the random_state attribute of an object if it exists.

        Parameters:
            obj: The object whose random_state attribute should be set.
        """
        if hasattr(obj, "random_state"):
            obj.random_state = self.random_state

    def _split(self, X, y):
        """
        Split the data into training and validation sets.

        Parameters:
        - X: The input features.
        - y: The target variable.

        Returns:
        - A generator that yields the indices of the training and validation sets.
        """
        split_y = (
            pd.qcut(y, self.num_y_quantiles, labels=False)
            if self.num_y_quantiles
            else y
        )
        return self.cv.split(X, split_y)

    def _check_inputs(self, X, y):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, pd.Series):
            y = y.values
        return X, y

    def cross_validate(self, X, y):
        """
        Fits the experiment model to the training data using cross-validation.

        Parameters:
            X: array-like, shape (n_samples, n_features)
            The input features.
            y: array-like, shape (n_samples,)
            The target variable.

        Returns:
            self: The fitted experiment model.
        """
        X, y = self._check_inputs(X, y)

        self.scores = []
        self.models = []
        self.oof_predictions = np.zeros_like(y)
        for train_index, test_index in self._split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train_preprocessed = self.preprocessor.fit_transform(X_train, y_train)
            self.estimator.fit(X_train_preprocessed, y_train)
            X_test_preprocessed = self.preprocessor.transform(X_test)
            self.models.append(self.estimator)
            self.oof_predictions[test_index] = self.estimator.predict(
                X_test_preprocessed
            )
            if self.score_func is not None:
                score = self.score_func(y_test, self.oof_predictions[test_index])
                self.scores.append(score)
        self.residuals = y - self.oof_predictions
        return self

    def fit(self, X, y):
        """
        Fits the experiment model to the training data.

        Parameters:
            X: array-like, shape (n_samples, n_features)
            The input features.
            y: array-like, shape (n_samples,)
            The target variable.

        Returns:
            self: The fitted experiment model.
        """
        X, y = self._check_inputs(X, y)
        self.estimator.fit(self.preprocessor.fit_transform(X, y), y)
        return self

    def predict(self, X):
        """
        Generates predictions using the trained model.

        Parameters:
            X: array-like, shape (n_samples, n_features)
            The input features.

        Returns:
            predictions: array-like, shape (n_samples,)
            The predicted target variable.
        """
        return self.estimator.predict(self.preprocessor.transform(X))

    def save(self, filename=None, root_dir=".") -> Path:
        """
        Save the current experiment to a file. If no filename is provided, a default
        filename will be used based on the object's name and the number of previous
        saves.

        Args:
            filename (str, optional): The name of the file to save the object to. If not
                provided, a default filename will be used based on the object's name.

        Returns:
            None
        """
        if filename is None:
            name = self.name.replace(" ", "_")
            previous_saves = glob.glob(f"{name}*", root_dir=root_dir)
            filename = name + f"_{len(previous_saves)}" + ".pkl"
        file_path = Path(root_dir) / filename
        joblib.dump(self, file_path)
        return file_path

    @staticmethod
    def load(filename):
        """
        Loads a saved experiment from a file.

        Parameters:
            filename: str
            The name of the file to load the experiment from.

        Returns:
            experiment: RegressionExperiment
            The loaded experiment object.
        """
        return joblib.load(filename)

    def __eq__(self, other):
        """
        Checks if two RegressionExperiment objects are equal.

        Parameters:
            other: RegressionExperiment
            The other RegressionExperiment object to compare.

        Returns:
            bool
            True if the two objects are equal, False otherwise.
        """
        if isinstance(other, RegressionExperiment):
            return (
                self.name == other.name
                and self.random_state == other.random_state
                and isinstance(self.cv, type(other.cv))
                and self.cv.get_n_splits() == other.cv.get_n_splits()
                and self.cv.random_state == other.cv.random_state
                and isinstance(self.preprocessor, type(other.preprocessor))
                and isinstance(self.estimator, type(other.estimator))
                and self.score_func == other.score_func
                and self.scores == other.scores
                and np.array_equal(self.oof_predictions, other.oof_predictions)
                and np.array_equal(self.residuals, other.residuals)
            )
        return False
