import datetime
import subprocess
import time

import prefect
from loguru import logger
from prefect import flow

from data_collection import QUERY_PATH
from training.tasks import (
    fetch_manga_data_task,
    generate_recommendations_task,
    load_data_task,
    preprocessing_task,
    train_baseline_models,
    train_knn_models,
    train_matrix_factorization_models,
)
from utils.bigquery_utils import write_data_to_bigquery
from utils.training_utils import get_best_model


def start_prefect_server():
    """
    Starts the Prefect server in the background if it's not already running.
    """
    try:
        # Check if the server is already running
        result = subprocess.run(
            ["pgrep", "-f", "prefect server start"], stdout=subprocess.PIPE
        )
        if result.stdout:
            logger.info("Prefect server is already running.")
            return

        # Start Prefect server
        logger.info("Starting Prefect server...")
        subprocess.Popen(
            ["prefect", "server", "start"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait a few seconds to ensure the server starts
        time.sleep(5)
        logger.info("Prefect server started successfully!")

    except Exception as e:
        logger.error(f"Failed to start Prefect server: {e}")


@flow(name="Fetching Manga Pipeline")
def fetch_manga_data_pipeline():
    """
    Prefect flow that automates manga data fetching and storing.
    """
    fetch_manga_data_task()


@flow(name="Training Pipeline")
def training_pipeline(query_path: str):
    """
    Prefect Flow to load data, preprocess, train, and evaluate recommender models.

    # TODO: Include preprocessing before training

    Args:
        query_path (str): Query path to fetch the dataset.
    """
    logger = prefect.get_run_logger()
    logger.info("Starting training recommender pipeline...")

    # Load the dataset
    df = load_data_task(query_path)

    # Preprocess the data
    df = preprocessing_task(df)

    # Train models in parallel
    matrix_future = train_matrix_factorization_models.submit(df)
    knn_future = train_knn_models.submit(df)
    baseline_future = train_baseline_models.submit(df)

    # Ensure all training tasks complete before proceeding
    matrix_future.result()
    knn_future.result()
    baseline_future.result()

    logger.success("Training Complete!")

    logger.info("Check workflow by running `prefect server start`")


@flow(
    name="Recommendation Pipeline", retries=0
)  # later: retries=1, retry_delay_seconds=5
def recommendation_pipeline(username: str, top_n: int = 10):
    """
    Prefect Flow to load data, train models, and make predictions.

    Args:
        query_path (str): Query path to fetch the dataset.
        username (str): Username for prediction.
        manga_title (str): Manga title for prediction.
    """
    logger = prefect.get_run_logger()
    logger.info("Starting manga recommendation pipeline...")

    # Load the dataset
    df = load_data_task(query_path)

    try:
        # Get the best model
        best_model = get_best_model(metric="mae", optimize="min")

        # Make predictions
        recommendations_df = generate_recommendations_task(
            username=username, model=best_model, df=df
        )

        logger.info(f"Top {top_n} Recommendations for {username}:")
        logger.info(recommendations_df.head(top_n))

        combined_df = load_data_task(query_path=f"{QUERY_PATH}/combine_users_manga.sql")

        write_data_to_bigquery(
            dataset_id="recommendations_dataset",
            table_id="recommendations",
            df=combined_df,
            mode="append",
        )

        logger.success(f"Recommendations for {username} completed successfully")

        logger.info("Check workflow by running `prefect server start`")

    except Exception as e:
        logger.error(f"Error in recommendation pipeline: {e}")
        raise


if __name__ == "__main__":
    query_path = f"{QUERY_PATH}/simple_train_data.sql"
    username = "Ruiisuuu"
    run_name = f"recommendations-for-{username}"

    start_prefect_server()

    # Run the training pipeline
    training_pipeline(query_path=query_path)

    # Get recommendations for a specific user
    recommendation_pipeline.with_options(name=run_name)(username=username)

    # activate prefect using prefect server start in terminal.
    # Then, open a new terminal and run workflow pipeline

    # Run the Prefect flow periodically using Prefect’s scheduling system
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import IntervalSchedule

    # Schedule the flow to run daily
    schedule = IntervalSchedule(interval=datetime.timedelta(hours=24))

    deployment = Deployment.build_from_flow(
        flow=fetch_manga_data_pipeline,
        name="fetch_manga_data_pipeline",
        schedule=schedule,
    )

    deployment.apply()
