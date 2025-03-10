import prefect
from prefect import flow

from training.tasks import (
    generate_recommendations,
    load_data,
    train_baseline_models,
    train_knn_models,
    train_matrix_factorization_models,
)
from utils.bigquery_utils import write_data_to_bigquery
from utils.training_utils import get_best_model


@flow(name="Training Pipeline")
def training_pipeline(query_path: str):
    """
    Prefect Flow to load data, train, and evaluate recommender models.

    Args:
        query_path (str): Query path to fetch the dataset.
    """
    logger = prefect.get_run_logger()
    logger.info("Starting training recommender pipeline...")

    # Load the dataset
    df = load_data(query_path)

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
    df = load_data(query_path)

    try:
        # Get the best model
        best_model = get_best_model(metric="mae", optimize="min")

        # Make predictions
        recommendations_df = generate_recommendations(
            username=username, model=best_model, df=df
        )

        logger.info(f"Top {top_n} Recommendations for {username}:")
        logger.info(recommendations_df.head(top_n))

        # Save dataframe to BigQuery
        write_data_to_bigquery(
            dataset_id="recommendations_dataset",
            table_id=f"recommendations_{username}",
            df=recommendations_df.reset_index(drop=True),
        )

        logger.info("Check workflow by running `prefect server start`")

    except Exception as e:
        logger.error(f"Error in recommendation pipeline: {e}")
        raise


if __name__ == "__main__":
    query_path = "./queries/simple_train_data.sql"
    username = "Ruiisuuu"
    run_name = f"recommendations-for-{username}"

    # Run the training pipeline
    training_pipeline(query_path=query_path)

    # Get recommendations for a specific user
    recommendation_pipeline.with_options(name=run_name)(username=username)

    # activate prefect using prefect server start in terminal.
    # Then, open a new terminal and run workflow pipeline
