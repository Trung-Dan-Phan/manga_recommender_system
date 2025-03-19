import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from preprocessing.feature_engineering import (
    encode_column,
    encode_genres,
    flag_reread,
    normalize_scores,
)


def test_normalize_scores():
    # Create a test DataFrame
    df = pd.DataFrame(
        {
            "score": [100, 10, 5, 3, np.nan],
            "score_format": ["POINT_100", "POINT_10", "POINT_5", "POINT_3", "POINT_10"],
        }
    )

    # Normalize scores
    result = normalize_scores(df)

    # Check normalization for each case
    assert result["normalized_score"][0] == 10  # 100 / 10 for POINT_100
    assert result["normalized_score"][1] == 10  # No change for POINT_10
    assert result["normalized_score"][2] == 10  # 5 * 2 for POINT_5
    assert result["normalized_score"][3] == 7  # 3 maps to 7.5 for POINT_3, rounded to 7
    assert np.isnan(result["normalized_score"][4])  # NaN should remain NaN


def test_encode_genres():
    # Create a test DataFrame
    df = pd.DataFrame(
        {
            "genres": [
                "Action, Adventure, Comedy",
                "Action, Romance, Drama",
                "Comedy, Romance",
                "Action, Comedy",
                "Action, Thriller",
            ]
        }
    )

    # Apply genre encoding with a minimum frequency of 2
    result = encode_genres(df, min_freq=2)

    # Check the resulting encoded columns
    assert "genre_Action" in result.columns
    assert "genre_Comedy" in result.columns
    assert "genre_Romance" in result.columns
    assert (
        "genre_Drama" not in result.columns
    )  # 'drama' should be dropped due to frequency
    assert result.shape[1] == 4  # 3 genres + the original genres column (after drop)

    # Check the actual encoding values for specific genres
    assert result["genre_Action"].sum() == 4  # "action" should be present in 4 rows
    assert result["genre_Comedy"].sum() == 3  # "comedy" should be present in 3 rows


def test_encode_column():
    # Create a test DataFrame
    df = pd.DataFrame({"status": ["Active", "Inactive", "Active", "Pending"]})

    # Encode the 'status' column using OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)
    result = encode_column(df, "status", encoder, prefix="status", drop_original=True)

    # Check that the original column is dropped
    assert "status" not in result.columns
    # Check that the encoded columns exist
    assert "status_Active" in result.columns
    assert "status_Inactive" in result.columns
    assert "status_Pending" in result.columns

    # Check encoding values
    assert result["status_Active"].sum() == 2
    assert result["status_Inactive"].sum() == 1
    assert result["status_Pending"].sum() == 1


def test_flag_reread():
    # Create a test DataFrame
    df = pd.DataFrame({"reread_count": [0, 1, 2, np.nan, 3]})

    # Flag rereads
    result = flag_reread(df)

    # Check that the 'reread_flag' column exists
    assert "reread_flag" in result.columns
    # Check the flag values
    assert result["reread_flag"][0] is False  # 0 rereads should be False
    assert result["reread_flag"][1] is True  # 1 reread should be True
    assert result["reread_flag"][2] is True  # 2 rereads should be True
    assert result["reread_flag"][3] is False  # NaN reread count should be False
    assert result["reread_flag"][4] is True  # 3 rereads should be True
