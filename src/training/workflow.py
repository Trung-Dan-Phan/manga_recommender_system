import pandas as pd
import prefect
from prefect import flow, task

from training import BASELINE_MODELS, KNN_MODELS, MATRIX_FACTORIZATION_MODELS
from training.predict import predict
from training.train import train_mlflow
from utils.bigquery_utils import load_data_from_bigquery
from utils.training_utils import get_best_model, save_recommendations


@task(name="Load Data", retries=3, retry_delay_seconds=5)
def load_data(
    query_path: str,
) -> pd.DataFrame:
    """
    Load dataset using a query saved in queries folder.

    Args:
        query (str): Query path to fetch dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    logger = prefect.get_run_logger()

    with open(query_path, "r", encoding="utf-8") as file:
        query = file.read().strip()

    df = load_data_from_bigquery(query=query)

    logger.info(f"Dataset loaded with {len(df)} rows.")
    return df


@task(name="Train Matrix Factorization Models", retries=2, retry_delay_seconds=10)
def train_matrix_factorization_models(df: pd.DataFrame):
    """Trains matrix factorization models (SVD, NMF, SlopeOne)."""
    for name, model in MATRIX_FACTORIZATION_MODELS.items():
        train_mlflow(df, model_name=name, model=model, similarity_metric="cosine")


@task(name="Train KNN Models", retries=2, retry_delay_seconds=10)
def train_knn_models(df: pd.DataFrame):
    """Trains KNN-based models (KNN Baseline, KNN Basic, etc.)."""
    for name, model in KNN_MODELS.items():
        train_mlflow(df, model_name=name, model=model, similarity_metric="cosine")


@task(name="Train Baseline Models", retries=2, retry_delay_seconds=10)
def train_baseline_models(df: pd.DataFrame):
    """Trains baseline models (Normal Predictor, Baseline Only, Co-clustering)."""
    for name, model in BASELINE_MODELS.items():
        train_mlflow(df, model_name=name, model=model, similarity_metric="cosine")


@task(name="Generate Recommendations", retries=2, retry_delay_seconds=5)
def generate_recommendations(
    username: str, best_model: str, df: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """
    Predicts ratings for all mangas for a given user and returns the top N recommendations.

    Args:
        username (str): The user to generate recommendations for.
        best_model (str): The name of the best performing model.
        df (pd.DataFrame): The dataset containing all mangas.
        top_n (int): Number of top recommendations to return.

    Returns:
        pd.DataFrame: Top N recommended mangas sorted by predicted rating.
    """
    unique_mangas = df["title_romaji"].unique()
    recommendations = []

    for manga_title in unique_mangas:
        predicted_rating = predict(best_model, username, manga_title)
        recommendations.append(
            {"manga_title": manga_title, "predicted_rating": round(predicted_rating, 2)}
        )

    # Convert to DataFrame and sort by predicted rating
    recommendations_df = pd.DataFrame(recommendations).sort_values(
        by="predicted_rating", ascending=False
    )

    save_recommendations(df=recommendations_df, username=username)

    # Return top N recommendations
    return recommendations_df.head(top_n)


@flow(
    name="Recommendation Pipeline", retries=0
)  # later: retries=1, retry_delay_seconds=5
def recommendation_pipeline(query_path: str, username: str, top_n: int = 10):
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

    # Train models in parallel
    matrix_future = train_matrix_factorization_models.submit(df)
    knn_future = train_knn_models.submit(df)
    baseline_future = train_baseline_models.submit(df)

    # Ensure all training tasks complete before proceeding
    matrix_future.result()
    knn_future.result()
    baseline_future.result()

    logger.info("Training Complete!")

    try:
        # Get the best model
        best_model = get_best_model(metric="MAE_mean", optimize="min")

        # Make predictions
        recommendations_df = generate_recommendations(
            username=username, best_model=best_model, df=df, top_n=top_n
        )

        logger.info(f"Top {top_n} Recommendations for {username}:")
        logger.info(recommendations_df.head(top_n))

        logger.info("Check workflow by running `prefect server start`")

    except Exception as e:
        logger.error(f"🔥 Error in recommendation pipeline: {e}")
        raise


if __name__ == "__main__":
    query_path = "./queries/simple_train_data.sql"
    username = "Ruiisuuu"
    run_name = f"recommendations-for-{username}"

    recommendation_pipeline.with_options(name=run_name)(
        query_path=query_path, username=username
    )
