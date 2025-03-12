import os
from typing import Optional

import pandas as pd
from google.cloud import bigquery
from loguru import logger
from prefect import task
from surprise import AlgoBase

from config import config  # Import your config.py
from data_collection import MANGA_DATASET_ID, fetch_manga_data
from preprocessing.clean_data import clean_data
from preprocessing.feature_engineering import feature_engineering
from training import BASELINE_MODELS, KNN_MODELS, MATRIX_FACTORIZATION_MODELS
from training.predict import generate_recommendations
from training.train import train_mlflow
from utils.bigquery_utils import load_data_from_bigquery, write_data_to_bigquery
from utils.fetching_utils import get_last_fetched_page

# Set the environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.GOOGLE_CREDENTIALS_PATH

# Initialize BigQuery client
client = bigquery.Client()


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


@task(name="Load Data from BigQuery", retries=3, retry_delay_seconds=5)
def load_data_task(
    dataset_id: Optional[str] = None,
    table_id: Optional[str] = None,
    query_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Loads data from BigQuery into a Pandas DataFrame.

    If a query is provided, it is used directly, and dataset_id and table_id are not required.
    Otherwise, both dataset_id and table_id must be provided to load the entire table.

    Args:
        dataset_id (Optional[str]): The ID of the BigQuery dataset. Required if query is None.
        table_id (Optional[str]): The table to fetch data from. Required if query is None.
        query_path (Optional[str]): Path to SQL query to execute.

    Returns:
        pd.DataFrame: The dataset loaded from BigQuery.
    """
    if query_path:
        with open(query_path, "r", encoding="utf-8") as file:
            query = file.read().strip()

    df = load_data_from_bigquery(dataset_id=dataset_id, table_id=table_id, query=query)
    return df


@task(name="Write Data to BigQuery", retries=3, retry_delay_seconds=5)
def write_data_task(
    dataset_id: str, table_id: str, df: pd.DataFrame, mode="overwrite"
) -> None:
    """
    Writes a DataFrame to a specified BigQuery table.

    Args:
        dataset_id (str): The ID of the BigQuery dataset.
        table_id (str): The name of the table where data should be written.
        df (pd.DataFrame): The Pandas DataFrame containing the data.
        mode (str): "append" to add new data or "overwrite" to replace existing data.

    Returns:
        None
    """
    write_data_to_bigquery(dataset_id=dataset_id, table_id=table_id, df=df, mode=mode)


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
