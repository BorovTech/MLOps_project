import pandas as pd
import numpy as np
from joblib import dump, load
from urllib.parse import urlparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import warnings
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted


def bootstrap_sample(batch: pd.DataFrame, n: int, k: int):
    """
    Function for bootstrapping a pandas df

    Args:
    batch - pandas df with data to be bootstrapped
    n - number of items in a single bootstrap sample
    k - number of samples to be returned

    Returns:
    Pandas dataframe with bootstrapped samples
    """

    assert n > 0, "Number of items in a sample must be greater than 0"
    assert len(batch) > 0, "length of the batch must be greater than 0"
    assert k > 0, "Number of samples must be greater than 0"
    df_list = []
    for i in range(k):
        sample = batch.sample(n=n, replace=True)
        df_list.append(sample)
    return pd.concat(df_list)


def concat_df(df1: pd.DataFrame, df2: pd.DataFrame, shuffle=True):
    """
    Function for concatenation of 2 dataframes
    pretty obvious ngl.....
    """
    dataframes = [df1, df2]
    assert all(
        [
            len(dataframes[0].columns.intersection(df.columns))
            == dataframes[0].shape[1]
            for df in dataframes
        ]
    ), "The dataframes' structure must be identical"
    if shuffle is False:
        return pd.concat(dataframes)
    else:
        c = pd.concat(dataframes)
        c = c.sample(frac=1)
        return c


def select_grid(model_name: str):
    """
    Function to select paramter grid for grid search depending on a model

    Args:
    model_name - name of the model. Must be one of these 4: 'XGBClassifier', "DecisionTreeClassifier", "LogisticRegression", 'RandomForestClassifier'
    Otherwise, raises an exception

    Returns:
    param_grid - dictionary with parameters for the model to be used in grid search
    """

    if model_name == "XGBClassifier":
        return {
            "learning_rate": [0.1, 0.01, 0.001],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
        }
    if model_name == "DecisionTreeClassifier":
        return {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        }
    if model_name == "LogisticRegression":
        return {"C": [0.001, 0.01, 0.1, 1, 10, 100], "penalty": ["none", "l2"]}
    if model_name == "RandomForestClassifier":
        return {
            "n_estimators": [25, 50, 100],
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 3, 5],
            "min_samples_split": [2, 5, 10],
        }
    raise Exception("Incorrect model name provided")


def refit(path_to_model: str, data: pd.DataFrame):
    """
    Function to fit a model on new data

    Args:
    path_to_model - path to a binary file with the model to be refitted
    data - pandas DataFrame with the data for the model to be refitted on

    Returns:
    best_model - the resulting best model selected with GridSearchCV

    Also, dumps the new best model to the same folder with the name best + model_name.joblib
    """

    assert (
        "y" in data.columns
    ), 'Target variable is not in the DataFrame, the name of target columns must be "y"'

    y_train = data["y"]
    X_train = data.drop("y", axis=1)
    model = load(path_to_model)
    model_name = type(model).__name__
    param_grid = select_grid(model_name)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="f1")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    check_is_fitted(best_model, msg="The model was not fitted :-(")
    dump(path_to_model, "best " + model_name + ".joblib")
    return best_model
