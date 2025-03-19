import pickle
import sys

import pandas as pd
from loguru import logger
from surprise import AlgoBase

from utils.bigquery_utils import load_data_from_bigquery


def load_model(model_name: str):
    """
    Load a trained recommendation model from a .pkl file.

    Parameters:
    - model_name (str): Name of the model to load (should match the saved filename).

    Returns:
    - model: The trained Surprise model instance.

    Raises:
    - FileNotFoundError: If the model file does not exist.
    """
    model_path = f"models/{model_name}.pkl"

    try:
        logger.info(f"Loading model '{model_name}' from {model_path}...")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model '{model_name}' successfully loaded.")
        return model
    except FileNotFoundError:
        logger.error(
            f"Model file '{model_path}' not found. Ensure the model is trained and saved correctly."
        )
        raise FileNotFoundError(
            f"Model '{model_name}' not found. Please check the file path."
        )


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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        logger.error(
            "Incorrect command usage."
            "Expected format: python -m training.predict <model_name> <username>"
        )
        sys.exit(1)

    model_name = sys.argv[1]
    username = sys.argv[2]

    # Load data from BigQuery
    query_file = "./queries/simple_train_data.sql"
    with open(query_file, "r", encoding="utf-8") as file:
        query = file.read().strip()

    df = load_data_from_bigquery(query=query)
    logger.info(f"Data shape: {df.shape}")

    # Load the model
    model = load_model(model_name)

    # Generate recommendations for given user and model name
    recommendations = generate_recommendations(
        username=username, model_name=model_name, df=df
    )
    logger.info(f"Recommendations for user '{username}'\n" f"{recommendations}': ")


def merge_recommendations_with_manga(
    recommendations_df: pd.DataFrame, manga_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges a recommendations DataFrame with a manga details DataFrame based `title_romaji`.

    Args:
        recommendations_df (pd.DataFrame): DataFrame containing user recommendations.
        manga_df (pd.DataFrame): DataFrame containing manga details.

    Returns:
        pd.DataFrame: A merged DataFrame containing user recommendations along with manga details,
                      ordered by `predicted_rating` in descending order.
    """
    # Filter columns
    recommendations_df = recommendations_df[
        ["username", "title_romaji", "predicted_rating"]
    ]

    # Perform the join on the 'title_romaji' column
    merged_df = recommendations_df.merge(manga_df, on="title_romaji", how="inner")

    # Remove duplicate entries and sort by predicted rating
    merged_df = merged_df.drop_duplicates().sort_values(
        by="predicted_rating", ascending=False
    )

    return merged_df
