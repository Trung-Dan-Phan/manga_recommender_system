import pickle
import sys

import pandas as pd
from loguru import logger


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


def predict(model_name: str, username: str, title_romaji: str) -> float:
    """
    Predict the rating a user would give to a manga using a trained recommendation model.

    Parameters:
    - model_name (str): Name of the trained model.
    - username (str): The user for whom we want to predict the rating.
    - title_romaji (str): The manga title for which the rating is predicted.

    Returns:
    - float: The estimated rating.

    Raises:
    - ValueError: If the model fails to make a prediction.
    """
    try:
        # Load the model
        model = load_model(model_name)

        # Make prediction
        prediction = model.predict(username, title_romaji)

        return prediction.est

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise ValueError(
            "Prediction could not be completed. Please check inputs and model validity."
        )


def generate_recommendations(
    username: str, best_model: str, df: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """
    Predicts ratings for all mangas and returns the top N recommendations.
    """
    try:
        unique_mangas = df["title_romaji"].unique()
        recommendations = []

        for manga_title in unique_mangas:
            predicted_rating = predict(best_model, username, manga_title)
            recommendations.append(
                {
                    "manga_title": manga_title,
                    "predicted_rating": round(predicted_rating, 2),
                }
            )

        recommendations_df = pd.DataFrame(recommendations).sort_values(
            by="predicted_rating", ascending=False
        )

        if recommendations_df.empty:
            raise ValueError("Recommendations DataFrame is empty!")

        logger.info(f"✅ Generated Top {top_n} Recommendations for {username}")
        return recommendations_df.head(top_n)

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise  # Ensure error stops retries


if __name__ == "__main__":
    if len(sys.argv) != 4:
        logger.error(
            "Incorrect command usage."
            "Expected format: python -m training.predict <model_name> <username> <title_romaji>"
        )
        print(
            "Usage: python -m training.predict <model_name> <username> <title_romaji>"
        )
        sys.exit(1)

    model_name = sys.argv[1]
    username = sys.argv[2]
    title_romaji = sys.argv[3]

    try:
        predicted_rating = predict(model_name, username, title_romaji)
        logger.info(
            f"Predicted rating for user '{username}' "
            f"and manga '{title_romaji} '"
            f"using model '{model_name}': "
            f"{predicted_rating:.2f}"
        )
    except ValueError:
        logger.warning("Prediction failed. Check logs for details.")

# Note: The prediction works well.
# However, we need to check why I still obtain a prediction if the user was not in the dataset
# and if we predict a score for a manga that does not exist.
# The idea would be to compute scores for all manga available in the dataset.
# Then do a ranking (sort by score) depending on manga attributes (e.g. genre)
