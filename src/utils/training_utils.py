import os
from typing import List, Optional, Tuple

import mlflow
import pandas as pd
from loguru import logger
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from surprise import AlgoBase, Dataset, dump


def split_dataframe(df: pd.DataFrame, test_size: float = 0.2, random_seed: int = 42):
    """
    Splits a DataFrame into train and test sets using stratification,
    while ensuring that users with only one rating are kept in the train set.

    Parameters:
    - df (pd.DataFrame): The full dataset containing (user_id, item_id, rating) columns.
    - test_size (float): Proportion of the dataset to include in the test set.
    - random_seed (int): Seed for reproducibility.

    Returns:
    - train_df (pd.DataFrame): Training set.
    - test_df (pd.DataFrame): Test set.
    """

    # Count the number of ratings per user
    user_counts = df["username"].value_counts()

    # Users with only one rating
    single_rating_users = user_counts[user_counts == 1].index
    multi_rating_users = user_counts[user_counts > 1].index

    # Ensure single-rating users are always in the training set
    df_single_ratings = df[df["username"].isin(single_rating_users)]
    df_multi_ratings = df[df["username"].isin(multi_rating_users)]

    # Perform stratified train/test split on users with multiple ratings
    train_df, test_df = train_test_split(
        df_multi_ratings,
        test_size=test_size,
        stratify=df_multi_ratings["username"],
        random_state=random_seed,
    )

    # Add users with only one rating to the train set
    train_df = pd.concat([train_df, df_single_ratings])

    logger.info(f"train shape: {train_df.shape}, test shape: {test_df.shape}")

    save_train_test_sets(train_df=train_df, test_df=test_df)

    return train_df, test_df


def save_train_test_sets(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Saves the training and test datasets to disk as CSV files.

    Parameters:
    - train_df (pd.DataFrame): Training dataset.
    - test_df (pd.DataFrame): Test dataset.

    Returns:
    - tuple: (train_df_path, test_df_path) where:
        - train_df_path (str): Path to the saved training set file.
        - test_df_path (str): Path to the saved test set file.

    """
    # Ensure the save directory exists
    save_dir = "models/data"
    os.makedirs(save_dir, exist_ok=True)

    # Define file paths
    train_df_path = os.path.join(save_dir, "train.csv")
    test_df_path = os.path.join(save_dir, "test.csv")

    # Save DataFrames as CSV
    train_df.to_csv(train_df_path, index=False)
    test_df.to_csv(test_df_path, index=False)

    return train_df_path, test_df_path


def save_model(
    model: AlgoBase,
    model_name: str,
    predictions: Optional[List] = None,
    verbose: int = 0,
) -> None:
    """
    Saves the trained model using Surprise's dump module.

    Parameters:
    - model (AlgoBase): The trained collaborative filtering model.
    - model_name (str): The name of the model.
    - predictions (list): List of predictions made by the model.
    """
    model_path = f"models/{model_name}.dump"
    os.makedirs("models", exist_ok=True)
    dump.dump(model_path, predictions=predictions, algo=model, verbose=verbose)

    # Log artifact inside the existing run
    mlflow.log_artifact(model_path)


def load_model(model_name: str) -> Tuple[Optional[List], Optional[AlgoBase]]:
    """
    Loads a trained model and its predictions using Surprise's dump module.

    Parameters:
    - model_name (str): The name of the model to load.

    Returns:
    - Tuple[Optional[List], Optional[AlgoBase]]: List of predictions and the loaded model.
    """
    model_path = f"models/{model_name}.dump"
    if os.path.exists(model_path):
        predictions, algo = dump.load(model_path)
        return predictions, algo
    else:
        logger.warning(f"Model file {model_path} not found.")
        return None, None


def get_best_model(
    metric: str = "mae",
    optimize: str = "min",
) -> AlgoBase:
    """
    Selects the best model from MLflow based on a given metric
    and optimization criteria on the testset.

    Args:
        metric (str): Metric to optimize (e.g., "mae", "precision_at_10").
        optimize (str): "min" for lower-better metrics, "max" for higher-better metrics.

    Returns:
        AlgoBase: Best model, or None if no suitable models are found.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Manga_Recommender")

    if not experiment:
        logger.error("MLflow experiment 'Manga_Recommender' not found!")
        return None

    # Search for all runs in the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=None,  # No filtering applied at this stage
        order_by=[f"metrics.{metric}_test {'ASC' if optimize == 'min' else 'DESC'}"],
    )

    if not runs:
        logger.error(f"No trained models found with metric '{metric}_test'")
        return None

    # Get the best model's metadata
    best_run = runs[0]
    best_model_name = best_run.data.tags.get("model_name")
    model_run_id = best_run.info.run_id

    best_model = None

    if not best_model_name:
        logger.error("No 'model_name' found in MLflow tags for the best model!")

    # Load model from MLflow artifacts
    model_path = (
        f"mlruns/{experiment.experiment_id}/{model_run_id}/artifacts/"
        f"{best_model_name}.dump"
    )
    logger.info(f"Loading best model from: {model_path}")

    try:
        best_model = dump.load(model_path)[
            1
        ]  # dump.load() returns (predictions, model)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return None

    logger.info(
        f"Best Model Loaded: {best_model_name} "
        f"based on metric '{metric}' ({optimize})"
    )

    return best_model


def get_k_nearest_neighbors(model: AlgoBase, trainset: Dataset, id: str, k: int = 5):
    """Retrieve the k nearest neighbors of a user or item using the trained model.

    Args:
        model (AlgoBase): A trained surprise KNN-based algorithm.
        trainset (Dataset): The trainset used to fit the model.
        id (str): The username or manga title for which we want neighbors.
        k (int): The number of nearest neighbors to retrieve. Default is 5.

    Returns:
        List of raw IDs of the k nearest neighbors.
    """
    # Determine if the algorithm is user-based or item-based
    is_user_based = model.sim_options["user_based"]

    try:
        if is_user_based:
            # Convert raw user ID to inner ID
            inner_id = trainset.to_inner_uid(id)
            # Get k nearest user neighbors
            neighbors = model.get_neighbors(inner_id, k=k)
            # Convert back to raw user IDs
            neighbors_raw_ids = [
                trainset.to_raw_uid(inner_id) for inner_id in neighbors
            ]
        else:
            # Convert raw item ID to inner ID
            inner_id = trainset.to_inner_iid(id)
            # Get k nearest item neighbors
            neighbors = model.get_neighbors(inner_id, k=k)
            # Convert back to raw item IDs
            neighbors_raw_ids = [
                trainset.to_raw_iid(inner_id) for inner_id in neighbors
            ]

        return neighbors_raw_ids

    except ValueError:
        return f"Error: The ID '{id}' does not exist in the dataset."


if __name__ == "__main__":
    logger.info(type(get_best_model()))
