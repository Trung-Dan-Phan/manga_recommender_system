import mlflow
import pandas as pd
from loguru import logger


def get_best_model(metric="MAE_mean", optimize="min") -> str:
    """
    Selects the best model from MLflow.

    Args:
        metric (str): Metric to optimize.
        optimize (str): "min" for MAE, "max" for Precision@K.

    Returns:
        str: Best model name.

    Raises:
        ValueError: If no models are found.
    """
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("Manga_Recommender")

        if not experiment:
            raise ValueError("MLflow experiment 'Manga_Recommender' not found!")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if optimize == 'min' else 'DESC'}"],
        )

        if not runs:
            raise ValueError(f"No trained models found with metric '{metric}'")

        best_model_name = runs[0].data.tags.get("model_name")
        logger.info(f"Best Model Selected: {best_model_name} based on {metric}")

        return best_model_name

    except Exception as e:
        logger.error(f"Error selecting best model: {e}")
        raise  # Ensure error is raised to stop retries


def save_recommendations(df: pd.DataFrame, username: str):
    """Saves the recommendations to both JSON and CSV files."""
    recommendations_path = "./data/recommendations"
    json_path = f"{recommendations_path}/recommendations_{username}.json"
    csv_path = f"{recommendations_path}/recommendations_{username}.csv"

    # Save to json and csv
    df.to_json(json_path, orient="records", indent=4)
    df.to_csv(csv_path, index=False)

    logger.info(f"Recommendations saved to {recommendations_path}")
