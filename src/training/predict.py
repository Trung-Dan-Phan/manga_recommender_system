import pickle
import sys

import pandas as pd
from loguru import logger

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
    username: str, model: str, df: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """
    Predicts ratings for all mangas and returns the top N recommendations.
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

        if recommendations_df.empty:
            raise ValueError("Recommendations DataFrame is empty!")

        logger.info(f"Generated Top {top_n} Recommendations for {username}")
        return recommendations_df.head(top_n)

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

# Note: The prediction works well.
# However, we need to check why I still obtain a prediction if the user was not in the dataset
# and if we predict a score for a manga that does not exist.
# The idea would be to compute scores for all manga available in the dataset.
# Then do a ranking (sort by score) depending on manga attributes (e.g. genre)
