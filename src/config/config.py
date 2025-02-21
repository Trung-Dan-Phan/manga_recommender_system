from google.cloud import bigquery
from loguru import logger

# Set authentication manually
client = bigquery.Client.from_service_account_json("./src/config/manga-recommender-system-d0a04ad1ff87.json")

# Check authentication
datasets = list(client.list_datasets())  # List available datasets
logger.info("Datasets in project:", datasets)
