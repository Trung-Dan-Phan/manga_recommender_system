import numpy as np
import pandas as pd

from src.preprocessing.clean_data import (
    clean_text_columns,
    drop_high_missing_columns,
    standardize_column_names,
)


# Test drop_high_missing_columns
def test_drop_high_missing_columns():
    # Create a DataFrame for testing
    df = pd.DataFrame(
        {
            "A": [1, np.nan, np.nan, 4],
            "B": [1, 2, 3, 4],
            "C": [np.nan, np.nan, np.nan, np.nan],
            "D": [1, 2, 3, 4],
        }
    )

    # Test case: drop columns with more than 50% missing values
    result = drop_high_missing_columns(df, threshold=0.5)
    assert "A" in result.columns
    assert "C" not in result.columns
    assert "B" in result.columns
    assert "D" in result.columns

    # Test case: No columns should be dropped when threshold is 1
    result = drop_high_missing_columns(df, threshold=1)
    assert "A" in result.columns
    assert "C" in result.columns

    # Test case: Columns with any nan should be dropped
    result = drop_high_missing_columns(df, threshold=0)
    assert "A" not in result.columns
    assert "C" not in result.columns


# Test clean_text_columns
def test_clean_text_columns():
    # Create a DataFrame with text columns
    df = pd.DataFrame(
        {
            "Manga Title": ["  Naruto  ", " One Piece ", None],
            "Genres": [" Action ", "Adventure ", " Drama "],
            "Other Column": [1, 2, 3],
        }
    )

    result = clean_text_columns(df)

    # Test that text columns are cleaned
    assert result["Manga Title"][0] == "naruto"
    assert result["Manga Title"][1] == "one piece"
    assert result["Genres"][0] == "action"
    assert result["Genres"][1] == "adventure"

    # Test that non-text columns are unaffected
    assert result["Other Column"][0] == 1

    # Test with missing values
    assert result["Manga Title"][2] is None  # Should remain None


# Test standardize_column_names
def test_standardize_column_names():
    # Create a DataFrame with columns that need standardizing
    df = pd.DataFrame(
        columns=["Column One", "Column Two$%", " Column Three  ", "Column_Four"]
    )

    result = standardize_column_names(df)

    # Test that the column names are standardized
    assert result.columns[0] == "column_one"
    assert result.columns[1] == "column_two"
    assert result.columns[2] == "column_three"
    assert result.columns[3] == "column_four"

    # Test with an empty DataFrame
    df_empty = pd.DataFrame()
    result_empty = standardize_column_names(df_empty)
    assert result_empty.columns.tolist() == []
