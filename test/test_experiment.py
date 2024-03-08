import tempfile

import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.experiment import RegressionExperiment


@pytest.fixture
def regression_data():
    return make_regression(n_samples=100, n_features=10, random_state=42)


@pytest.fixture
def experiment():
    return RegressionExperiment(
        "Test Pipeline",
        KFold(n_splits=5),
        StandardScaler(),
        LinearRegression(),
        mean_squared_error,
    )


@pytest.fixture
def model(regression_data, experiment):
    return experiment.fit(*regression_data)


def test_oof_prediction(regression_data, model):
    _, y = regression_data
    predictions = model.oof_predictions
    assert predictions.shape == y.shape


def test_save_load_with_filename(model):
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = "test_pipeline.pkl"
        file_path = model.save(filename, temp_dir)
        loaded_pipeline = RegressionExperiment.load(file_path)
        assert model == loaded_pipeline


def test_save_load_without_filename(model):
    with tempfile.TemporaryDirectory() as root_dir:
        file1 = model.save(root_dir=root_dir)
        file2 = model.save(root_dir=root_dir)
        assert file1 != file2
