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
from utils import *

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
    origin_path = Path(f"Models/src/better_{model_name}.joblib")
    return load(origin_path)


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
    return df.sample(batch_size)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    BATCH_SIZE = 100
    df = pd.read_csv("Models/data/val.csv")
    data = get_batch(df, BATCH_SIZE)
    X = data.drop("y", axis=1)
    y = data["y"]

    with mlflow.start_run():
        MODEL_NAME = input("Enter a model name: ")

        mlflow.set_tag("model_name", MODEL_NAME)
        # TODO: argparse the model name
        # Loading the model
        model = load_model(MODEL_NAME)

        # Making the predictions and comparing them with ground truth values
        preds = model.predict(X)
        acc, f1 = calc_metrics(preds, y)

        if (f1 < 0.75) or (acc < 0.75):
            print("THE MODEL NEEDS TO BE RETRAINED!!!")
            new_df = bootstrap_sample(data, int(BATCH_SIZE / 2), 5)
            new_df = concat_df(new_df, df)
            updated_model = refit(f"Models/src/better_{MODEL_NAME}.joblib", new_df)

        print(f"Model: {MODEL_NAME}")
        print(f"Accuracy score on batch of size {BATCH_SIZE} is: {acc}")
        print(f"F1-score on batch of size {BATCH_SIZE} is: {f1}")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1-score", f1)

        signature = infer_signature(X, preds)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        mlflow.sklearn.log_model(
            model, "model", registered_model_name=MODEL_NAME, signature=signature
        )
