import pandas as pd
import numpy as np
from joblib import dump, load
from urllib.parse import urlparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import mlflow
import warnings
from mlflow.models.signature import infer_signature
from pathlib import Path

# import argparse
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def calc_metrics(preds: np.ndarray, truth: np.ndarray) -> tuple:
    """
    A function that calculates a set of metrics, used in our production

    Arguments:
    preds (np.NDarray) - An array of predicted values
    truth (np.NDarray) - An array of ground truth values

    Returns:
    A tuple of calculated metrics
    """
    acc = accuracy_score(truth, preds)
    f1 = f1_score(truth, preds)
    return acc, f1


def load_model(model_name: str):
    """
    A function that loads a given model

    Args:
    path (str) - Path to a file with a model, presumably with .joblib extension

    Returns:
    An object of our model
    """
    origin_path = Path("Models/data/models/")
    path = origin_path / Path(model_name + ".joblib")
    return load(path)


def dump_model(model, path: str) -> None:
    """
    A function that dumps the given model to a specified file
    (the path to a file should exist!!!)

    Args:
    model - a model that we want to save
    path - a path to our model
    """
    dump(model, path)


def get_batch(df: pd.DataFrame, batch_size: int = 32):
    """
    A function that extracts a batch of data from a given DataFrame
    and utilises it in order to simulate real-world inference

    Args:
    df (pd.DataFrame) - A DataFrame to use
    batch_size (int) - A size of batch to extract from a DataFrame

    Returns:
    pd.DataFrame - a DataFrame of that batch only with the necessary features
    """
    list_of_features = [
        "month",
        "duration",
        "nr.employed",
        "poutcome",
        "emp.var.rate",
        "y",
    ]
    return df.sample(batch_size)[list_of_features]


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    BATCH_SIZE = 100
    df = pd.read_csv("Models/data/val.csv")
    data = get_batch(df, BATCH_SIZE)
    X = data.drop("y", axis=1)
    y = data["y"]

    with mlflow.start_run():
        # TODO: argparse the model name
        # Loading the model
        MODEL_NAME = "logreg"
        model = load_model(MODEL_NAME)

        # Making the predictions and comparing them with ground truth values
        preds = model.predict(X)
        acc, f1 = calc_metrics(preds, y)

        print(f"Model: {MODEL_NAME}")
        print(f"Accuracy score on batch of size {BATCH_SIZE} is: {acc}")
        print(f"F1-score on batch of size {BATCH_SIZE} is: {f1}")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1-score", f1)

        signature = infer_signature(X, preds)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=MODEL_NAME,
                signature=signature,
            )
        else:
            mlflow.sklearn.log_model(model, "model", signature=signature)
