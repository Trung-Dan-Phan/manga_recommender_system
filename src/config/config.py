import os

from google.cloud import bigquery
from google.oauth2 import service_account
from loguru import logger

# Absolute path to the service account JSON file
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)  # Get current directory of config.py
GOOGLE_CREDENTIALS_PATH = os.path.join(
    BASE_DIR, "manga-recommender-system-d0a04ad1ff87.json"
)

# Ensure the file exists
if not os.path.exists(GOOGLE_CREDENTIALS_PATH):
    raise FileNotFoundError(
        f"Service account file not found at {GOOGLE_CREDENTIALS_PATH}"
    )

# Load credentials
credentials = service_account.Credentials.from_service_account_file(
    GOOGLE_CREDENTIALS_PATH
)

# Initialize BigQuery client
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# Check authentication
try:
    # Fetch dataset names
    datasets = list(client.list_datasets())

    # Extract dataset IDs (names)
    dataset_names = [dataset.dataset_id for dataset in datasets]

    # Log the dataset names
    logger.info(f"Datasets in project: {dataset_names}")

except Exception as e:
    logger.error(f"Failed to list datasets: {e}")
