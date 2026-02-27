# Wine Quality Prediction with MLflow
# Fully compatible with MLflow 3.x

import sys
import warnings
import logging

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import mlflow
import mlflow.sklearn


# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------- Metrics Function ----------------
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


# ---------------- Main ----------------
if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # ✅ IMPORTANT: Set tracking server
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Wine_Quality_Experiment")

    # Dataset URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )

    # Load dataset
    try:
        data = pd.read_csv(csv_url, sep=";")
        print("Dataset loaded successfully ✅")
    except Exception as e:
        logger.exception("Dataset load failed: %s", e)
        sys.exit(1)

    # Train-test split
    train, test = train_test_split(data, test_size=0.25, random_state=42)

    train_x = train.drop("quality", axis=1)
    test_x = test.drop("quality", axis=1)
    train_y = train["quality"]
    test_y = test["quality"]

    # ElasticNet hyperparameters from CLI
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # Models dictionary
    models = {
        "LinearRegression": LinearRegression(),
        "SVR": SVR(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "ElasticNet": ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, random_state=42
        ),
    }

    # Train and log models
    for model_name, model in models.items():

        with mlflow.start_run(run_name=model_name):

            # Train
            model.fit(train_x, train_y)

            # Predict
            predictions = model.predict(test_x)

            # Evaluate
            rmse, mae, r2 = eval_metrics(test_y, predictions)

            print(f"\n===== {model_name} =====")
            print(f"RMSE: {rmse}")
            print(f"MAE : {mae}")
            print(f"R2  : {r2}")

            # Log parameters
            mlflow.log_param("model_name", model_name)

            if model_name == "ElasticNet":
                mlflow.log_param("alpha", alpha)
                mlflow.log_param("l1_ratio", l1_ratio)

            # Log metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # ✅ Log model (NO registry to avoid error)
            mlflow.sklearn.log_model(model, name="model")

    print("\n🎉 All models trained and logged successfully!")