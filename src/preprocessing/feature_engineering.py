import re
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

from utils.bigquery_utils import load_data_from_bigquery, write_data_to_bigquery


def sanitize_column_name(name: str) -> str:
    """
    Sanitize a string to be used as a column name.
    Replaces any non-word character with an underscore.

    Parameters:
        name (str): Original string.

    Returns:
        str: Sanitized string.
    """
    sanitized = re.sub(r"[^\w]", "_", name)
    return sanitized


def normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the 'score' column based on the 'score_format' column,
    creating a new column 'normalized_score' on a 10-point scale using vectorized operations.

    The transformation is applied by filtering the DataFrame for each score format:
      - "POINT_100": The score is divided by 10.
      - "POINT_10" or "POINT_10_DECIMAL": The score remains unchanged.
      - "POINT_5": The score is multiplied by 2.
      - "POINT_3": The score is mapped as follows: 1 -> 2.5, 2 -> 5, 3 -> 7.5.
        Other values will be set as NaN.
    Then, we round the scores to the nearest integer.

    Parameters:
        df (pd.DataFrame): DataFrame with 'score' and 'score_format' columns.

    Returns:
        pd.DataFrame: DataFrame with an added column 'normalized_score'.
    """
    logger.info("Normalizating scores to a 10-point scale.")

    # Initialize the new column with NaN values
    df["normalized_score"] = np.nan

    # For "POINT_100": Divide the score by 10
    mask = (df["score_format"] == "POINT_100") & (df["score"].notna())
    df.loc[mask, "normalized_score"] = df.loc[mask, "score"] / 10
    logger.debug(f"Normalized {mask.sum()} scores for POINT_100.")

    # For "POINT_10" and "POINT_10_DECIMAL": No change needed
    mask = df["score_format"].isin(["POINT_10", "POINT_10_DECIMAL"]) & (
        df["score"].notna()
    )
    df.loc[mask, "normalized_score"] = df.loc[mask, "score"]
    logger.debug(f"Assigned {mask.sum()} scores for POINT_10/POINT_10_DECIMAL.")

    # For "POINT_5": Multiply the score by 2
    mask = (df["score_format"] == "POINT_5") & (df["score"].notna())
    df.loc[mask, "normalized_score"] = df.loc[mask, "score"] * 2
    logger.debug(f"Normalized {mask.sum()} scores for POINT_5.")

    # For "POINT_3": Map 1->2.5, 2->5, 3->7.5
    mask = (df["score_format"] == "POINT_3") & (df["score"].notna())
    mapping = {1: 2.5, 2: 5, 3: 7.5}
    df.loc[mask, "normalized_score"] = df.loc[mask, "score"].map(mapping)
    logger.debug(f"Normalized {mask.sum()} scores for POINT_3.")

    # Round the normalized scores to the nearest integer (between 1 and 10)
    df["normalized_score"] = df["normalized_score"].round().clip(1, 10).astype(int)

    logger.info("Completed normalization of scores.")
    return df


def encode_genres(df: pd.DataFrame, min_freq: int = 10) -> pd.DataFrame:
    """
    Encodes the 'genres' column using multi-label encoding.
    Only genres appearing at least `min_freq` times are retained.

    Parameters:
        df (pd.DataFrame): DataFrame with a 'genres' column.
        min_freq (int): Minimum frequency for a genre to be kept.

    Returns:
        pd.DataFrame: DataFrame with new genre indicator columns appended.
    """
    logger.info("Encoding 'genres' column using multi-label encoding.")
    df["genres_list"] = (
        df["genres"]
        .fillna("")
        .apply(lambda x: [i.strip() for i in x.split(",")] if x else [])
    )
    genre_counts = df["genres_list"].explode().value_counts()
    frequent_genres = genre_counts[genre_counts >= min_freq].index.tolist()
    logger.info(
        f"Retaining {len(frequent_genres)} genres with frequency >= {min_freq}."
    )
    df["genres_list"] = df["genres_list"].apply(
        lambda lst: [genre for genre in lst if genre in frequent_genres]
    )

    mlb = MultiLabelBinarizer()
    genres_encoded = pd.DataFrame(
        mlb.fit_transform(df["genres_list"]),
        columns=[f"genre_{sanitize_column_name(genre)}" for genre in mlb.classes_],
        index=df.index,
    )
    logger.debug(f"Encoded genres shape: {genres_encoded.shape}")
    df = pd.concat([df, genres_encoded], axis=1)
    df.drop(columns=["genres_list"], inplace=True)
    logger.info("Completed encoding 'genres' column.")
    return df


def encode_column(
    df: pd.DataFrame,
    column_name: str,
    encoder,
    prefix: Optional[str] = None,
    drop_original: bool = False,
) -> pd.DataFrame:
    """
    Encode a column in a DataFrame using a provided encoder.

    This function applies the specified encoder (e.g., OneHotEncoder, OrdinalEncoder)
    to the column with name `column_name`. The encoded result is converted into a DataFrame
    with new column names (prefixed by `prefix` if provided, or the original column name otherwise)
    and concatenated with the original DataFrame. Optionally, the original column can be dropped.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to encode.
        encoder: An encoder instance that implements the fit_transform method.
        prefix (Optional[str]): Prefix for the encoded columns. Defaults to the column name.
        drop_original (bool): If True, the original column is dropped after encoding.

    Returns:
        pd.DataFrame: The DataFrame with the encoded features appended.

    Example usage:
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False)
        df = encode_column(df, "status", encoder, prefix="status", drop_original=True)
    """
    logger.info(
        f"Encoding column '{column_name}' using {encoder.__class__.__name__} encoder."
    )

    # Use the provided column as a DataFrame (required for scikit-learn encoders)
    col_data = df[[column_name]]

    # Fit and transform the data
    encoded_array = encoder.fit_transform(col_data)

    # Convert sparse matrices (if any) to dense arrays
    if hasattr(encoded_array, "toarray"):
        encoded_array = encoded_array.toarray()

    # Set the prefix for encoded columns
    if prefix is None:
        prefix = column_name

    # Try to use the encoder's get_feature_names_out method if available
    # otherwise, use default naming
    try:
        feature_names = encoder.get_feature_names_out([prefix])
    except Exception as e:
        logger.debug(f"Encoder does not support get_feature_names_out: {e}")
        feature_names = [f"{prefix}_{i}" for i in range(encoded_array.shape[1])]

    # Create a DataFrame for the encoded columns
    encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
    logger.debug(f"Encoded '{column_name}' into columns: {encoded_df.columns.tolist()}")

    # Concatenate the new columns to the original DataFrame
    df = pd.concat([df, encoded_df], axis=1)

    # Optionally drop the original column
    if drop_original:
        df = df.drop(columns=[column_name])
        logger.info(f"Dropped original column '{column_name}' after encoding.")

    logger.info(
        f"Completed encoding for column '{column_name}' into {encoded_df.shape[1]} features."
    )
    return df


def flag_reread(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a boolean flag indicating whether a manga was reread at least once.

    Parameters:
        df (pd.DataFrame): DataFrame with a 'reread_count' column.

    Returns:
        pd.DataFrame: DataFrame with a new boolean column 'reread_flag'.
    """
    logger.info("Flagging entries with rereads.")
    df["reread_flag"] = df["reread_count"].apply(
        lambda x: True if pd.notna(x) and x > 0 else False
    )
    logger.debug(f"First 5 'reread_flag' values: {df['reread_flag'].head().tolist()}")
    logger.info("Completed flagging reread entries.")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform comprehensive feature engineering on the manga reading dataset.

    This function applies the following operations:
      - Normalize scores to a 10-point scale.
      - Encode the 'genres' column via multi-label encoding.
      - One-hot encode 'status' and 'format' columns.
      - Process the 'custom_lists' column to count the number of lists.
      - Calculate the reading duration from 'started_reading' and 'completed_reading'.
      - Calculate the number of days since 'started_reading'.
      - Create a flag for reread entries.

    Parameters:
        df (pd.DataFrame): Input DataFrame with columns:
                           'username', 'manga_title', 'genres', 'format', 'score',
                           'progress', 'status', 'custom_lists', 'started_reading',
                           'completed_reading', 'reread_count', 'score_format'.

    Returns:
        pd.DataFrame: DataFrame with new engineered features.
    """
    logger.info("Starting feature engineering for manga reading dataset.")

    # Normalize scores
    df = normalize_scores(df)

    # Encode genres with a frequency filter
    if "genres" in df.columns:
        df = encode_genres(df, min_freq=10)

    # One-hot encode 'status' column using the new encode_column function
    if "status" in df.columns:
        ohe_status = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        df = encode_column(
            df, "status", ohe_status, prefix="status", drop_original=True
        )

    # One-hot encode 'format' column using the new encode_column function
    if "format" in df.columns:
        ohe_format = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        df = encode_column(
            df, "format", ohe_format, prefix="format", drop_original=True
        )

    # Flag reread entries
    if "reread_count" in df.columns:
        df = flag_reread(df)

    logger.info("Feature engineering completed.")
    return df


if __name__ == "__main__":
    # Load data from BigQuery
    df = load_data_from_bigquery(
        dataset_id="users_dataset", table_id="user_manga_list_clean"
    )
    logger.info(f"Data shape (pre feature engineering): {df.shape}")

    # Perform feature engineering
    df_engineered = feature_engineering(df)

    logger.info(f"Data shape (post feature engineering): {df_engineered.shape}")

    # Save or inspect the engineered DataFrame
    write_data_to_bigquery(
        dataset_id="users_dataset",
        table_id="user_manga_list_processed",
        df=df_engineered,
    )

    logger.info("Feature engineering completed!")
