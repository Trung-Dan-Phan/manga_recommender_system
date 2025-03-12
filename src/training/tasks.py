import pandas as pd
import prefect
from loguru import logger
from prefect import task
from surprise import AlgoBase

from data_collection import MANGA_DATASET_ID, fetch_manga_data
from preprocessing.clean_data import clean_data
from preprocessing.feature_engineering import feature_engineering
from training import BASELINE_MODELS, KNN_MODELS, MATRIX_FACTORIZATION_MODELS
from training.predict import generate_recommendations
from training.train import train_mlflow
from utils.bigquery_utils import load_data_from_bigquery, write_data_to_bigquery
from utils.fetching_utils import get_last_fetched_page


@task(name="Fetch Manga Data")
def fetch_manga_data_task():
    dataset_id = MANGA_DATASET_ID
    table_id = "manga_collection_raw"

    # Get last fetched manga ID
    last_fetched_page = get_last_fetched_page(dataset_id=dataset_id, table_id=table_id)
    logger.info(f"Last fetched page: {last_fetched_page}")

    # Fetch new manga data with a max limit of 2000
    new_manga_data = fetch_manga_data(start_page=last_fetched_page + 1)

    if new_manga_data:
        new_manga_df = pd.DataFrame(new_manga_data)

        # Append to BigQuery
        write_data_to_bigquery(
            dataset_id=dataset_id, table_id=table_id, df=new_manga_df, mode="append"
        )

        logger.info(f"Saved {len(new_manga_df)} new manga records!")

    else:
        logger.info("No new manga found.")


@task(name="Load Data", retries=3, retry_delay_seconds=5)
def load_data_task(
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


@task(name="Preprocessing", retries=3, retry_delay_seconds=5)
def preprocessing_task(df: pd.DataFrame):
    """
    Performs data cleaning, feature engineering, and handling missing values.
    """
    # Perform data cleaning
    df_clean = clean_data(df)

    # Perform feature engineering
    df_engineered = feature_engineering(df_clean)

    return df_engineered


@task(name="Train Matrix Factorization Models", retries=2, retry_delay_seconds=10)
def train_matrix_factorization_models(df: pd.DataFrame):
    """Trains matrix factorization models (SVD, NMF, SlopeOne)."""
    for model_name, model in MATRIX_FACTORIZATION_MODELS.items():
        train_mlflow(
            df=df, model_name=model_name, model=model, similarity_metric="cosine"
        )


@task(name="Train KNN Models", retries=2, retry_delay_seconds=10)
def train_knn_models(df: pd.DataFrame):
    """Trains KNN-based models (KNN Baseline, KNN Basic, etc.)."""
    for model_name, model in KNN_MODELS.items():
        train_mlflow(
            df=df, model_name=model_name, model=model, similarity_metric="cosine"
        )


@task(name="Train Baseline Models", retries=2, retry_delay_seconds=10)
def train_baseline_models(df: pd.DataFrame):
    """Trains baseline models (Normal Predictor, Baseline Only, Co-clustering)."""
    for model_name, model in BASELINE_MODELS.items():
        train_mlflow(
            df=df, model_name=model_name, model=model, similarity_metric="cosine"
        )


@task(name="Generate Recommendations", retries=2, retry_delay_seconds=5)
def generate_recommendations_task(
    username: str, model: AlgoBase, df: pd.DataFrame
) -> pd.DataFrame:
    """
    Predicts ratings for all mangas and returns the top N recommendations.

    Parameters:
    - username (str): Username for whom recommendations are to be generated.
    - model (AlgoBase): Trained recommendation model.
    - df (pd.DataFrame): DataFrame containing user-manga ratings.

    Returns:
    - pd.DataFrame: DataFrame containing top N recommendations for the given user.
    """
    recommendations_df = generate_recommendations(username=username, model=model, df=df)

    return recommendations_df
