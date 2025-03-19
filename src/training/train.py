import os
import time
from typing import List

import mlflow
import mlflow.sklearn
import pandas as pd
from loguru import logger
from surprise import (
    AlgoBase,
    Dataset,
    KNNBaseline,
    KNNBasic,
    KNNWithMeans,
    KNNWithZScore,
    Reader,
    Trainset,
)
from surprise.accuracy import fcp, mae, rmse
from surprise.model_selection.split import KFold
from surprise.model_selection.validation import cross_validate

from training import BASELINE_MODELS, KNN_MODELS, MATRIX_FACTORIZATION_MODELS
from training.evaluate import precision_recall_at_k
from utils.bigquery_utils import load_data_from_bigquery
from utils.training_utils import save_model


def prepare_dataset(df: pd.DataFrame) -> Dataset:
    """
    Loads the dataset into Surprise's format.

    Parameters:
    - df (pd.DataFrame): The dataset containing user ratings.

    Returns:
    - Dataset: Surprise dataset object.
    """
    reader = Reader(rating_scale=(1, 10))
    return Dataset.load_from_df(df, reader)


def configure_knn_model(
    model: AlgoBase, similarity_metric: str, user_based: bool
) -> None:
    """
    Configures the similarity options for KNN-based models.

    Parameters:
    - model (AlgoBase): The collaborative filtering model.
    - similarity_metric (str): The similarity metric to use.
    - user_based (bool): Whether to use user-based or item-based filtering.
    """
    if isinstance(model, (KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore)):
        model.sim_options = {"name": similarity_metric, "user_based": user_based}


def save_dataset_info(
    data: Trainset, model_name: str, similarity_metric: str, k: int, threshold: int
) -> str:
    """
    Extracts and saves dataset statistics as a JSON file.

    Parameters:
    - data (Trainset): The dataset containing user ratings.
    - model_name (str): The name of the model.
    - similarity_metric (str): The similarity metric used.
    - k (int): Number of recommendations to consider.
    - threshold (int): The rating threshold for relevance.

    Returns:
    - str: Path to the saved dataset info JSON file.
    """
    # Extract dataset statistics
    dataset_info = {
        "num_users": data.n_users,
        "num_items": data.n_items,
        "num_ratings": data.n_ratings,
        "rating_scale": data.rating_scale,
        "global_mean": data.global_mean,
        "similarity_metric": similarity_metric,
        "k": k,
        "threshold": threshold,
    }

    # Save dataset info JSON file
    os.makedirs("models", exist_ok=True)
    dataset_info_path = f"models/data/{model_name}_dataset_info.json"
    pd.DataFrame([dataset_info]).to_json(dataset_info_path)

    return dataset_info_path


def train(
    model: AlgoBase,
    data: Dataset,
    accuracy_measures: List[str],
    k: int,
    threshold: int,
    cv: int = 5,
    n_jobs: int = 1,
) -> None:
    """
    Performs K-Fold cross-validation, computes Precision@K and Recall@K.

    Parameters:
    - model (AlgoBase): The collaborative filtering model.
    - data (Dataset): Surprise dataset object.
    - accuracy_measures (List[str]): List of accuracy measures to compute.
    - k (int): Number of recommendations to consider.
    - threshold (int): The rating threshold for relevance.
    - cv (int, optional): Number of folds for cross-validation. Defaults to 5.
    - n_jobs (int, optional): Number of jobs to run in parallel. Defaults to 1.
    """
    # Evaluate accuracy measures
    results = cross_validate(
        algo=model,
        data=data,
        measures=accuracy_measures,
        cv=cv,
        verbose=True,
        n_jobs=n_jobs,
    )

    # Log metrics
    for metric in accuracy_measures:
        mlflow.log_metric(f"{metric.lower()}_train", results[f"test_{metric}"].mean())

    # Evaluate Precision and Recall @k
    kf = KFold(n_splits=cv, random_state=0)
    precision_list, recall_list = [], []

    start_time = time.time()

    for train_set, validation_set in kf.split(data):
        model.fit(train_set)
        predictions = model.test(validation_set)
        precisions, recalls = precision_recall_at_k(
            predictions, k=k, threshold=threshold, log=False
        )
        precision_list.append(
            sum(prec for prec in precisions.values()) / len(precisions)
        )
        recall_list.append(sum(rec for rec in recalls.values()) / len(recalls))

    avg_precision = sum(precision_list) / len(precision_list) if precision_list else 0
    avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0
    train_time = time.time() - start_time

    logger.info(
        f"Average Precision@{k}: {avg_precision:.4f}, "
        f"Average Recall@{k}: {avg_recall:.4f} "
        f"in {round(train_time, 2)}sec"
    )

    # Log metrics
    mlflow.log_metric(f"training_time_at_{k}", train_time)
    mlflow.log_metric(f"precision_at_{k}_train", avg_precision)
    mlflow.log_metric(f"recall_at_{k}_train", avg_recall)


def evaluate(
    model: AlgoBase,
    testset: Dataset,
    k: int,
    threshold: int,
) -> None:
    """
    Evaluates the model on the test set.

    Parameters:
    - model (AlgoBase): The collaborative filtering model.
    - testset (Dataset): Surprise testing dataset.
    - k (int): Number of recommendations to consider.
    - threshold (int): The rating threshold for relevance.
    """
    # Evaluate accuracy measures
    predictions = model.test(testset=testset)

    # Compute RMSE & MAE
    RMSE_test = rmse(predictions, verbose=False)
    MAE_test = mae(predictions, verbose=False)
    FCP_test = fcp(predictions, verbose=False)

    logger.info(
        f"RMSE_test: {RMSE_test:.4f}, MAE_test: {MAE_test:.4f}, FCP_test: {FCP_test:.4f}"
    )

    precision_test, recall_test = precision_recall_at_k(
        predictions, k=k, threshold=threshold, log=False
    )

    # Compute averages
    avg_precision_test = (
        sum(precision_test.values()) / len(precision_test) if precision_test else 0
    )
    avg_recall_test = sum(recall_test.values()) / len(recall_test) if recall_test else 0

    logger.info(
        f"Average Precision@{k}_test: {avg_precision_test:.4f}, "
        f"Average Recall@{k}_test: {avg_recall_test:.4f}"
    )

    # Log metrics
    mlflow.log_metric("rmse_test", RMSE_test)
    mlflow.log_metric("mae_test", MAE_test)
    mlflow.log_metric("fcp_test", FCP_test)
    mlflow.log_metric(f"precision_at_{k}_test", avg_precision_test)
    mlflow.log_metric(f"recall_at_{k}_test", avg_recall_test)


def train_mlflow(
    df: pd.DataFrame,
    model_name: str,
    model: AlgoBase,
    similarity_metric: str = "msd",
    user_based: bool = True,
    accuracy_measures: List[str] = ["rmse", "mae", "fcp"],
    k: int = 10,
    threshold: int = 7,
) -> None:
    """
    Main function to train the model, compute evaluation metrics, and log everything in MLflow.

    Parameters:
    - df (pd.DataFrame): The dataset containing user ratings.
    - model_name (str): The name of the model.
    - model (AlgoBase): The collaborative filtering model.
    - similarity_metric (str): The similarity metric used.
    - user_based (bool): Whether user-based filtering is used.
    - accuracy_measures (List[str]): List of accuracy measures to compute.
    - k (int): Number of recommendations to consider.
    - threshold (int): The rating threshold for relevance.
    """
    # Ensure no active MLflow run exists before starting a new one
    if mlflow.active_run():
        logger.debug("mlflow run has ended")
        mlflow.end_run()

    # Prepare datasets
    data = prepare_dataset(df)
    trainset = data.build_full_trainset()
    testset = trainset.build_testset()

    # Configure KNN Models
    configure_knn_model(
        model=model, similarity_metric=similarity_metric, user_based=user_based
    )

    # Save dataset info
    dataset_info_path = save_dataset_info(
        data=trainset,
        model_name=model_name,
        similarity_metric=similarity_metric,
        k=k,
        threshold=threshold,
    )

    mlflow.set_experiment("Manga_Recommender")

    try:
        with mlflow.start_run(run_name=model_name):
            logger.info(
                f"Training {model_name} with Precision@{k} and Recall@{k} evaluation..."
            )

            # Log dataset artifacts
            mlflow.log_artifact(dataset_info_path)

            # Log tags
            mlflow.set_tags(
                {
                    "algorithm_type": type(model).__name__,
                    "model_name": model_name,
                    "similarity_metric": similarity_metric,
                    "user_based": user_based,
                }
            )

            # Train model
            train(
                model=model,
                data=data,
                accuracy_measures=accuracy_measures,
                k=k,
                threshold=threshold,
            )

            # Evaluate model on test set
            evaluate(
                model=model,
                testset=testset,
                k=k,
                threshold=threshold,
            )

            # Save and log model
            save_model(model=model, model_name=model_name)

            logger.info(f"{model_name} training completed.")

    except Exception as e:
        logger.error(f"Error during MLflow run: {e}")

    finally:
        if mlflow.active_run():
            mlflow.end_run()


if __name__ == "__main__":
    # Load data from BigQuery
    query_file = "./queries/simple_processed_users.sql"
    with open(query_file, "r", encoding="utf-8") as file:
        query = file.read().strip()

    df = load_data_from_bigquery(query=query)
    logger.info(f"Data shape: {df.shape}")

    # Train and log each model
    MODELS = {}
    MODELS.update(BASELINE_MODELS)
    MODELS.update(KNN_MODELS)
    MODELS.update(MATRIX_FACTORIZATION_MODELS)

    for model_name, model in MODELS.items():
        train_mlflow(
            df=df, model_name=model_name, model=model, similarity_metric="cosine"
        )

    logger.info("Training completed!")
    logger.info("See the results by starting `mlflow ui` in your terminal")

    # TODO: Check how can we add additional features
