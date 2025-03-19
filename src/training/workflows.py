import subprocess
import time

import prefect
import psutil
from loguru import logger
from prefect import flow

from data_collection import MANGA_DATASET_ID, QUERY_PATH
from training.tasks import (
    fetch_manga_data_task,
    generate_recommendations_task,
    load_data_task,
    merge_recommendations_with_manga_task,
    preprocessing_task,
    train_baseline_models,
    train_knn_models,
    train_matrix_factorization_models,
    write_data_task,
)
from utils.training_utils import get_best_model


def start_prefect_server():
    """
    Starts the Prefect server in the background if it's not already running.
    """
    try:
        # Check if the server is already running
        for process in psutil.process_iter(attrs=["pid", "name"]):
            if "prefect" in process.info["name"]:
                logger.info("Prefect server is already running.")
                return

        # Start Prefect server
        logger.info("Starting Prefect server...")
        subprocess.run("start cmd /k prefect server start", shell=True, check=True)

        # Wait a few seconds to ensure the server starts
        time.sleep(5)
        logger.info("Prefect server started successfully!")

    except Exception as e:
        logger.error(f"Failed to start Prefect server: {e}")


@flow(name="Fetching Manga Pipeline")
def fetch_manga_data_workflow():
    """
    Prefect flow that automates manga data fetching and storing.
    """
    fetch_manga_data_task()


@flow(name="Training Pipeline")
def training_workflow(query_path: str):
    """
    Prefect Flow to load data, preprocess, train, and evaluate recommender models.

    Args:
        query_path (str): Query path to fetch the dataset.
    """
    logger = prefect.get_run_logger()
    logger.info("Starting training recommender pipeline...")

    # Load the dataset
    df = load_data_task(query_path)

    # Preprocess the data
    df = preprocessing_task(df)

    # Select features for training

    df = df[["username", "title_romaji", "normalized_score"]]

    # Train models in parallel
    matrix_future = train_matrix_factorization_models.submit(df)
    knn_future = train_knn_models.submit(df)
    baseline_future = train_baseline_models.submit(df)

    # Ensure all training tasks complete before proceeding
    matrix_future.result()
    knn_future.result()
    baseline_future.result()

    logger.info("Training Complete!")

    logger.info("Check workflow by running `prefect server start`")


@flow(
    name="Recommendation Pipeline", retries=0
)  # later: retries=1, retry_delay_seconds=5
def recommendation_workflow(query_path: str, username: str, top_n: int = 10):
    """
    Prefect Flow to load data, train models, and make predictions.

    Args:
        query_path (str): Query path to fetch the processed dataset.
        username (str): Username for prediction.
        manga_title (str): Manga title for prediction.
    """
    logger = prefect.get_run_logger()
    logger.info("Starting manga recommendation pipeline...")

    # Load the training dataset
    train_df = load_data_task(query_path=query_path)

    try:
        # Get the best model
        best_model = get_best_model(metric="mae", optimize="min")

        # Make predictions
        recommendations_df = generate_recommendations_task(
            username=username, model=best_model, df=train_df
        )

        logger.info(f"Top {top_n} Recommendations for {username}:")
        logger.info(recommendations_df.head(top_n))

        # Load manga dataset
        manga_df = load_data_task(
            dataset_id=MANGA_DATASET_ID, table_id="manga_collection_clean"
        )

        # Combine recommendations with manga data
        combined_df = merge_recommendations_with_manga_task(
            recommendations_df=recommendations_df, manga_df=manga_df
        )

        # Overwrite recommendations in BigQuery
        write_data_task(
            dataset_id="recommendations_dataset",
            table_id="recommendations",
            df=combined_df,
            mode="append",
        )

        logger.info(f"Recommendations for {username} completed successfully")

        logger.info("Check workflow by running `prefect server start`")

    except Exception as e:
        logger.error(f"Error in recommendation pipeline: {e}")
        raise


if __name__ == "__main__":
    training_query_path = f"{QUERY_PATH}/get_positive_scores.sql"
    recommendation_query_path = f"{QUERY_PATH}/simple_processed_users.sql"
    username = "Ruiisuuu"
    run_name = f"recommendations-for-{username}"

    start_prefect_server()

    # Register flows for deployment
    fetch_manga_data_workflow.deploy(
        name="fetch-manga-data",
        work_pool_name="data-fetch-pool",
        image="trungdan/prefect-recommender-system-fetching:latest",
        cron="0 9 * * *",  # every day at 9am
        build=False,
    )

    training_workflow.deploy(
        name="train-model",
        parameters={"query_path": training_query_path},
        work_pool_name="model-train-pool",
        image="trungdan/prefect-recommender-system-training:latest",
        cron="0 9 * * 1",  # every Monday at 9am
        build=False,
    )

    recommendation_workflow.serve(
        name="generate-recommendations",
        parameters={"query_path": recommendation_query_path, "username": username},
    )

    # NOTE: Since models are large, we need to store them in cloud storage (e.g. S3)
    # To load them using a Docker image.
