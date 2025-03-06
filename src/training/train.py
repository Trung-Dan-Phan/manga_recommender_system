import os
import pickle
import time

import mlflow
import mlflow.sklearn
import pandas as pd
from loguru import logger
from surprise import (
    Dataset,
    KNNBaseline,
    KNNBasic,
    KNNWithMeans,
    KNNWithZScore,
    Reader,
)
from surprise.model_selection import KFold, cross_validate

from training import BASELINE_MODELS, KNN_MODELS, MATRIX_FACTORIZATION_MODELS
from training.evaluate import precision_recall_at_k
from utils.bigquery_utils import load_data_from_bigquery


def train_mlflow(
    df: pd.DataFrame,
    model_name: str,
    model,
    similarity_metric: str = "msd",
    user_based: bool = True,
    accuracy_measures: list = ["rmse", "mae", "fcp"],
):
    """
    Train a given recommendation model, log results in MLflow, and save the trained model.

    Parameters:
    - df (pd.Dataframe): The training dataset (username, title, score).
    - model_name (str): The name of the model.
    - model: The scikit-surprise model instance.
    - similarity_metric (str): Similarity metric for KNN-based models
    ('cosine', 'msd', 'pearson', 'pearson_baseline').
    - user_based (bool): Whether to perform user-based or item-based collaborative filtering.
    - accuracy_measures (list): List of accuracy metrics to evaluate
    ('rmse', 'mse', 'mae', 'fcp').
    """
    # Prepare dataset
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(df, reader)

    # Build the full trainset
    trainset = data.build_full_trainset()

    # Configure similarity for KNN models
    if isinstance(model, (KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore)):
        model.sim_options = {"name": similarity_metric, "user_based": user_based}

    # Extract dataset statistics
    num_users = len(set(df["username"]))
    num_items = len(set(df["title_romaji"]))
    num_ratings = len(df)

    # Dataset info to be saved as artifacts
    dataset_info = {
        "num_users": num_users,
        "num_items": num_items,
        "num_ratings": num_ratings,
        "similarity_metric": similarity_metric,
        "accuracy_measures": accuracy_measures,
    }
    dataset_info_path = f"models/{model_name}_dataset_info.json"
    os.makedirs("models", exist_ok=True)
    pd.DataFrame([dataset_info]).to_json(dataset_info_path)

    # Save the dataset as an artifact
    dataset_path = f"models/{model_name}_dataset.csv"
    df.to_csv(dataset_path, index=False)

    # Set MLflow experiment
    mlflow.set_experiment("Manga_Recommender")

    try:
        with mlflow.start_run(run_name=model_name):
            logger.info(f"Training {model_name} with {similarity_metric} similarity...")

            # Add tags to the MLflow run
            mlflow.set_tags(
                {
                    "algorithm_type": type(model).__name__,
                    "model_name": model_name,
                    "similarity_metric": similarity_metric,
                    "user_based": user_based,
                }
            )

            # Log dataset artifacts
            mlflow.log_artifact(dataset_path)
            mlflow.log_artifact(dataset_info_path)

            # Log dataset details
            mlflow.log_param("num_users", num_users)
            mlflow.log_param("num_items", num_items)
            mlflow.log_param("num_ratings", num_ratings)
            mlflow.log_param("similarity_metric", similarity_metric)
            mlflow.log_param("user_based", user_based)

            # Log model-specific hyperparameters (if applicable)
            if hasattr(model, "n_factors"):  # SVD, NMF
                mlflow.log_param("n_factors", model.n_factors)
            if hasattr(model, "k"):  # KNN models
                mlflow.log_param("k", model.k)

            # Start timing
            start_time = time.time()

            # Fit the model on the full dataset
            model.fit(trainset)

            # Measure training time
            train_time = time.time() - start_time
            mlflow.log_metric("training_time", train_time)

            # Perform cross-validation (evaluate using specified metrics)
            results = cross_validate(
                model, data, measures=accuracy_measures, cv=5, verbose=True
            )

            # Log selected metrics
            for metric in accuracy_measures:
                mlflow.log_metric(
                    f"{metric.upper()}_mean", results[f"test_{metric}"].mean()
                )

            # Save model
            model_path = f"models/{model_name}.pkl"
            os.makedirs("models", exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            mlflow.log_artifact(model_path)

            logger.info(f"{model_name} training completed and logged in MLflow.")

    except Exception as e:
        logger.error(f"Error during MLflow run: {e}")
    finally:
        # Ensure the MLflow run is closed
        mlflow.end_run()


def train_mlflow_with_precision_recall(
    df: pd.DataFrame,
    model_name: str,
    model,
    similarity_metric: str = "msd",
    user_based: bool = True,
    k: int = 10,
    threshold: int = 7,
):
    """
    Train a model, compute Precision@K and Recall@K, and log results in MLflow.

    Parameters:
    - df (pd.Dataframe): The training dataset (username, title, score).
    - model_name (str): The name of the model.
    - model: The scikit-surprise model instance.
    - similarity_metric (str): Similarity metric for KNN-based models
    ('cosine', 'msd', 'pearson', 'pearson_baseline').
    - user_based (bool): Whether to perform user-based or item-based collaborative filtering.
    - k (int): Number of recommendations to consider for Precision@K and Recall@K.
    - threshold (int): Rating threshold for relevance.
    """
    # Prepare dataset
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(df, reader)

    # Build the full trainset
    trainset = data.build_full_trainset()

    # Configure similarity for KNN models
    if isinstance(model, (KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore)):
        model.sim_options = {"name": similarity_metric, "user_based": user_based}

    # Extract dataset statistics
    num_users = len(set(df["username"]))
    num_items = len(set(df["title_romaji"]))
    num_ratings = len(df)

    # Dataset info to be saved as artifacts
    dataset_info = {
        "num_users": num_users,
        "num_items": num_items,
        "num_ratings": num_ratings,
        "similarity_metric": similarity_metric,
        "k": k,
        "threshold": threshold,
    }
    dataset_info_path = f"models/{model_name}_dataset_info.json"
    os.makedirs("models", exist_ok=True)
    pd.DataFrame([dataset_info]).to_json(dataset_info_path)

    # Save the dataset as an artifact
    dataset_path = f"models/{model_name}_dataset.csv"
    df.to_csv(dataset_path, index=False)

    # Set MLflow experiment
    mlflow.set_experiment("Manga_Recommender")

    try:
        with mlflow.start_run(run_name=model_name):
            logger.info(
                f"Training {model_name} with Precision@{k} and Recall@{k} evaluation..."
            )

            # Add tags to the MLflow run
            mlflow.set_tags(
                {
                    "algorithm_type": type(model).__name__,
                    "model_name": model_name,
                    "similarity_metric": similarity_metric,
                    "user_based": user_based,
                }
            )

            # Log dataset artifacts
            mlflow.log_artifact(dataset_path)
            mlflow.log_artifact(dataset_info_path)

            # Log dataset details
            mlflow.log_param("num_users", num_users)
            mlflow.log_param("num_items", num_items)
            mlflow.log_param("num_ratings", num_ratings)
            mlflow.log_param("similarity_metric", similarity_metric)
            mlflow.log_param("user_based", user_based)

            # Log model-specific hyperparameters (if applicable)
            if hasattr(model, "n_factors"):  # SVD, NMF
                mlflow.log_param("n_factors", model.n_factors)
            if hasattr(model, "k"):  # KNN models
                mlflow.log_param("k", model.k)

            # K-Fold Cross-Validation
            kf = KFold(n_splits=5, random_state=42)
            precision_list, recall_list = [], []

            # Start timing
            start_time = time.time()

            for trainset, testset in kf.split(data):
                # Train the model
                model.fit(trainset)
                predictions = model.test(testset)

                # Compute Precision@K and Recall@K
                precisions, recalls = precision_recall_at_k(
                    predictions, k=k, threshold=threshold
                )

                # Aggregate results
                precision_list.append(
                    sum(prec for prec in precisions.values()) / len(precisions)
                )
                recall_list.append(sum(rec for rec in recalls.values()) / len(recalls))

            # Compute average Precision@K and Recall@K
            precision_avg = sum(precision_list) / len(precision_list)
            recall_avg = sum(recall_list) / len(recall_list)

            # Measure training time
            train_time = time.time() - start_time
            mlflow.log_metric("training_time", train_time)

            logger.info(
                f"Precision@{k}: {precision_avg:.4f}, Recall@{k}: {recall_avg:.4f}"
            )

            # Log metrics in MLflow
            mlflow.log_metric(f"Precision_at_{k}", precision_avg)
            mlflow.log_metric(f"Recallat_at_{k}", recall_avg)

            # Save model
            model_path = f"models/{model_name}.pkl"
            os.makedirs("models", exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            mlflow.log_artifact(model_path)

            logger.info(f"{model_name} training completed.")

    except Exception as e:
        logger.error(f"Error during MLflow run: {e}")
    finally:
        # Ensure the MLflow run is closed
        mlflow.end_run()


if __name__ == "__main__":
    # Load data from BigQuery
    query_file = "./queries/simple_train_data.sql"
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
        train_mlflow_with_precision_recall(
            df=df, model_name=model_name, model=model, similarity_metric="cosine"
        )

    logger.info("Training completed!")
    logger.info("See the results by starting `mlflow ui` in your terminal")

    # TODO: Check how can we add additional features
