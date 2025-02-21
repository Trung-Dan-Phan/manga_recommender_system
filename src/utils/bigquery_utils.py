from google.cloud import bigquery
import pandas as pd
from loguru import logger

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
    except Exception as e:
        logger.warning(f"Dataset '{dataset_id}' not found. Creating a new dataset...")
        dataset = bigquery.Dataset(f"{client.project}.{dataset_id}")
        dataset.location = "US"  # Change location if necessary
        client.create_dataset(dataset, exists_ok=True)
        logger.info(f"Dataset '{dataset_id}' created successfully.")

def load_data_from_bigquery(dataset_id: str, table_id: str) -> pd.DataFrame:
    """
    Loads a dataset from BigQuery into a Pandas DataFrame.

    Args:
        dataset_id (str): The ID of the BigQuery dataset.
        table_id (str): The name of the table to fetch data from.

    Returns:
        pd.DataFrame: The dataset loaded from BigQuery.
    """
    TABLE_REF = f"{client.project}.{dataset_id}.{table_id}"
    query = f"SELECT * FROM `{TABLE_REF}`"
    
    try:
        df = client.query(query).to_dataframe()
        logger.info(f"Successfully loaded {len(df)} rows from {TABLE_REF}")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset from BigQuery: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error

def write_data_to_bigquery(dataset_id: str, table_id: str, csv_path: str):
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

    # Read CSV into Pandas DataFrame
    df = pd.read_csv(csv_path)

    TABLE_REF = f"{client.project}.{dataset_id}.{table_id}"

    try:
        job = client.load_table_from_dataframe(df, TABLE_REF)
        job.result()  # Wait for completion
        logger.info(f"Successfully uploaded {len(df)} rows to {TABLE_REF}")
    except Exception as e:
        logger.error(f"Failed to write CSV to BigQuery: {e}")
