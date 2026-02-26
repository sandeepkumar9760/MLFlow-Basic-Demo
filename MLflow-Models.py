# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties.
# In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys
import logging

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from urllib.parse import urlparse
import mlflow
import mlflow.sklearn


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read dataset
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )

    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download CSV. Check internet connection. Error: %s", e
        )

    # Train-test split
    train, test = train_test_split(data, test_size=0.25, random_state=42)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train["quality"]
    test_y = test["quality"]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # Set tracking URI (comment this if running locally)
    # remote_server_uri = "http://ec2-54-147-36-34.compute-1.amazonaws.com:5000/"
    # mlflow.set_tracking_uri(remote_server_uri)

    models = {
        "LinearRegression": LinearRegression(),
        "SVR": SVR(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "ElasticNet": ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42),
    }

    for model_name, model in models.items():

        with mlflow.start_run(run_name=model_name):

            model.fit(train_x, train_y)

            predictions = model.predict(test_x)

            rmse, mae, r2 = eval_metrics(test_y, predictions)

            print(f"\n===== {model_name} =====")
            print("RMSE:", rmse)
            print("MAE:", mae)
            print("R2:", r2)

            mlflow.log_param("model_name", model_name)

            if model_name == "ElasticNet":
                mlflow.log_param("alpha", alpha)
                mlflow.log_param("l1_ratio", l1_ratio)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=f"{model_name}WineModel",
                )
            else:
                mlflow.sklearn.log_model(model, "model")