import os
from typing import Optional

import pandas as pd
from google.cloud import bigquery
from loguru import logger

from config import config  # Import your config.py

# Set the environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.GOOGLE_CREDENTIALS_PATH

# Initialize BigQuery client
client = bigquery.Client()


def ensure_dataset_exists(dataset_id: str):
    """
    Check if a BigQuery dataset exists. If not, create it.

    Args:
        dataset_id (str): The dataset ID in BigQuery.

    Returns:
        None
    """
    dataset_ref = client.dataset(dataset_id)

    try:
        client.get_dataset(dataset_ref)  # Check if dataset exists
        logger.info(f"Dataset '{dataset_id}' already exists.")
    except Exception:
        logger.warning(
            f"Dataset '{dataset_id}' not found. " "Creating a new dataset..."
        )
        dataset = bigquery.Dataset(f"{client.project}.{dataset_id}")
        dataset.location = "US"  # Change location if necessary
        client.create_dataset(dataset, exists_ok=True)
        logger.info(f"Dataset '{dataset_id}' created successfully.")


def load_data_from_bigquery(
    dataset_id: Optional[str] = None,
    table_id: Optional[str] = None,
    query: Optional[str] = None,
) -> pd.DataFrame:
    """
    Loads data from BigQuery into a Pandas DataFrame.

    Args:
        dataset_id (Optional[str]): ID of the BigQuery dataset. Required if no query is provided.
        table_id (Optional[str]): Name of the BigQuery table. Required if no query is provided.
        query (Optional[str]): SQL query. If provided, dataset_id and table_id are ignored.

    Returns:
        pd.DataFrame: The dataset loaded from BigQuery.

    Example usage:
    ```
    # Using a query
    query_file = '../queries/query.sql'
    with open(query_file, "r", encoding="utf-8") as file:
        query = file.read().strip()

    df = load_data_from_bigquery(query=query)

    # Loading an entire table
    df = load_data_from_bigquery(dataset_id="my_dataset", table_id="my_table")
    ```
    """
    try:
        if query is None:
            if not dataset_id or not table_id:
                raise ValueError(
                    "Both dataset_id and table_id must be provided if no query is given."
                )

            # Load entire table
            table_ref = f"{client.project}.{dataset_id}.{table_id}"
            query = f"SELECT * FROM `{table_ref}`"

        logger.info(f"Loading data using query: {query}")

        df = client.query(query).to_dataframe()
        logger.info(f"Successfully loaded {len(df)} rows from BigQuery")
        return df

    except Exception as e:
        logger.error(f"Failed to load dataset from BigQuery: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error


def write_data_to_bigquery(dataset_id: str, table_id: str, df: pd.DataFrame):
    """
    Writes a CSV file to a specified BigQuery table.

    Args:
        dataset_id (str): The ID of the BigQuery dataset.
        table_id (str): The name of the table where data should be written.
        csv_path (str): The local path to the CSV file.

    Returns:
        None
    """
    ensure_dataset_exists(dataset_id)  # Ensure dataset exists before writing

    TABLE_REF = f"{client.project}.{dataset_id}.{table_id}"

    # Overwrite table if TABLE_REF already exists
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
    )

    try:
        job = client.load_table_from_dataframe(df, TABLE_REF, job_config=job_config)
        job.result()  # Wait for completion
        logger.info(f"Successfully uploaded {len(df)} rows to {TABLE_REF}")
    except Exception as e:
        logger.error(f"Failed to write CSV to BigQuery: {e}")
