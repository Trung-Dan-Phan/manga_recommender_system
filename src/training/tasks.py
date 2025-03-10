import pandas as pd
import prefect
from loguru import logger
from prefect import task
from surprise import AlgoBase

from training import BASELINE_MODELS, KNN_MODELS, MATRIX_FACTORIZATION_MODELS
from training.train import train_mlflow
from utils.bigquery_utils import load_data_from_bigquery


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
def generate_recommendations(
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
    try:
        unique_mangas = df["title_romaji"].unique()
        recommendations = []

        for manga_title in unique_mangas:
            # Make prediction
            prediction = model.predict(username, manga_title)
            # Append recommendation with prediction parameters
            recommendations.append(
                {
                    "uid": prediction.uid,  # User ID
                    "iid": prediction.iid,  # Manga Title (Item ID)
                    "r_ui": prediction.r_ui,  # True rating (if available, else None)
                    "est": int(prediction.est),  # Predicted rating
                    "details": prediction.details,  # Additional details
                }
            )

        recommendations_df = pd.DataFrame(recommendations).sort_values(
            by="est", ascending=False
        )

        # Extract 'details' dictionary into separate columns
        recommendations_df = pd.concat(
            [
                recommendations_df.drop(columns=["details"]),
                recommendations_df["details"].apply(pd.Series),
            ],
            axis=1,
        )

        # Rename columns for better readability
        recommendations_df.rename(
            columns={
                "uid": "username",
                "iid": "title_romaji",
                "r_ui": "actual_rating",
                "est": "predicted_rating",
            },
            inplace=True,
        )

        if recommendations_df.empty:
            raise ValueError("Recommendations DataFrame is empty!")

        logger.info(f"Generated Recommendations for {username}")
        return recommendations_df

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise  # Ensure error stops retries
