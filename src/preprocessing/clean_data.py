import re

import pandas as pd
from loguru import logger

from utils.bigquery_utils import load_data_from_bigquery, write_data_to_bigquery


def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Drop columns where the fraction of missing values is above the threshold.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        threshold (float): The maximum allowed fraction of missing values for a column.

    Returns:
        pd.DataFrame: The DataFrame with high-missing columns removed.
    """
    logger.info("Dropping columns with missing values above threshold...")
    missing_frac = df.isnull().mean()
    cols_to_drop = missing_frac[missing_frac > threshold].index.tolist()
    logger.info(f"Columns to drop: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)
    return df


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean text-based columns by trimming whitespace, converting text to lowercase,
    and removing unwanted characters.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with cleaned text columns.
    """
    logger.info("Cleaning text columns...")
    text_cols = ["Manga Title", "Genres", "Format", "Status", "Custom Lists"]
    for col in text_cols:
        if col in df.columns:
            # Preserve None/NaN values while cleaning the text
            df[col] = df[col].apply(
                lambda x: str(x).strip().lower() if pd.notna(x) else x
            )
            logger.debug(f"Cleaned column: {col}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with duplicates removed.
    """
    logger.info("Removing duplicate rows...")
    initial_count = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_count - len(df)} duplicate rows.")
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to ensure consistency and compatibility with downstream processes.
    For instance, convert all column names to lowercase and replace spaces or special characters.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with standardized column names.
    """
    logger.info("Standardizing column names...")
    df.columns = [
        re.sub(r"[^\w]+", "_", col.strip().lower()).rstrip("_") for col in df.columns
    ]
    logger.debug(f"New column names: {df.columns.tolist()}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main cleaning function that applies all cleaning operations to the DataFrame.

    The cleaning steps include:
      - Dropping columns with high missing values.
      - Cleaning text columns.
      - Converting date columns to datetime objects.
      - Removing duplicate rows.
      - Standardizing column names.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    logger.info("Starting data cleaning process...")
    df = drop_high_missing_columns(df, threshold=0.7)
    df = clean_text_columns(df)
    df = remove_duplicates(df)
    df = standardize_column_names(df)
    logger.info("Data cleaning process completed.")
    return df


if __name__ == "__main__":
    # Load data from BigQuery
    query_file = "./queries/get_positive_scores.sql"
    with open(query_file, "r", encoding="utf-8") as file:
        query = file.read().strip()

    df = load_data_from_bigquery(query=query)
    logger.info(f"Loaded data with shape: {df.shape}")

    # Perform data cleaning
    df_clean = clean_data(df)
    logger.info(f"Cleaned data shape: {df_clean.shape}")

    # Save or inspect the engineered DataFrame
    write_data_to_bigquery(
        dataset_id="users_dataset", table_id="user_manga_list_clean", df=df_clean
    )
    logger.info("Data cleaning completed!")
