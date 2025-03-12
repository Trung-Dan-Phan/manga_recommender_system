import os

from google.api_core.exceptions import NotFound
from google.cloud import bigquery
from loguru import logger

from config import config  # Import your config.py

# Set the environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.GOOGLE_CREDENTIALS_PATH

# Initialize BigQuery client
client = bigquery.Client()


def get_last_fetched_page(dataset_id: str, table_id: str):
    """
    Retrieves the last fetched page number from BigQuery.
    Returns 0 if the table does not exist or is empty.

    Args:
        dataset_id (str): The BigQuery dataset ID.
        table_id (str): The BigQuery table name storing metadata.

    Returns:
        int: The last fetched page number, or 0 if the table is empty or doesn't exist.
    """
    TABLE_REF = f"{client.project}.{dataset_id}.{table_id}"

    query = f"""
        SELECT MAX(page) AS last_page
        FROM `{TABLE_REF}`
    """

    try:
        query_job = client.query(query)
        results = query_job.result()

        last_fetched_page = 0  # Default to 0 if no record exists
        for row in results:
            if row.last_page:
                last_fetched_page = row.last_page
            break

        return last_fetched_page

    except NotFound:
        logger.warning(f"Table `{TABLE_REF}` does not exist. Returning 0.")
        return 0  # Return 0 if table does not exist

    except Exception as e:
        logger.error(f"Failed to fetch last fetched page: {e}")
        return 0  # Return 0 on any other unexpected error
