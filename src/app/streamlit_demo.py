import os

import pandas as pd
import requests
import streamlit as st
from google.cloud import bigquery

from config import config  # Import your config.py
from data_collection import RECOMMENDATIONS_DATASET_ID, USERS_DATASET_ID

# Set the environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.GOOGLE_CREDENTIALS_PATH

# Initialize BigQuery client
client = bigquery.Client()

# Constants
TABLE_RECOMMENDATIONS = "recommendations"
TABLE_USERS = "user_manga_list_raw"
PREFECT_WORKFLOW_URL = (
    "http://127.0.0.1:4200/api/deployments/"
    "6e868437-442b-4bae-be41-67579e2dcc52/create_flow_run"
)


# Function to check if recommendations exist
def get_recommendations(username: str) -> pd.DataFrame:
    query = f"""
    SELECT *
    FROM `{client.project}.{RECOMMENDATIONS_DATASET_ID}.{TABLE_RECOMMENDATIONS}`
    WHERE username = '{username}'
    ORDER BY predicted_rating DESC
    """
    query_job = client.query(query)
    return query_job.to_dataframe()


# Function to check if the user exists in users dataset
def check_user_exists(username: str) -> bool:
    query = f"""
    SELECT username
    FROM `{client.project}.{USERS_DATASET_ID}.{TABLE_USERS}`
    WHERE username = '{username}'
    """
    query_job = client.query(query)
    df = query_job.to_dataframe()
    return not df.empty


# Function to trigger Prefect workflow
def trigger_recommendation_workflow(username: str) -> bool:
    """
    Triggers the Prefect workflow for generating manga recommendations.

    Args:
        username (str): The username for which recommendations should be generated.

    Returns:
        bool: True if the workflow was successfully triggered, False otherwise.
    """
    payload = {
        "parameters": {
            "query_path": "./queries/simple_processed_users.sql",
            "username": username,
            "top_n": 10,
        }
    }

    try:
        response = requests.post(PREFECT_WORKFLOW_URL, json=payload)

        if response.status_code in [200, 201]:
            print("Workflow triggered successfully!")
            return True
        else:
            print(f"Failed to trigger workflow. Status code: {response.status_code}")
            print(response.text)
            return False

    except requests.exceptions.RequestException as e:
        print(f"Error triggering workflow: {e}")
        return False


# Streamlit UI
st.title("Manga Recommender Demo")
st.write("By Trung Dan Phan [GitHub](https://github.com/Trung-Dan-Phan)")
st.markdown("[Source code](https://github.com/Trung-Dan-Phan/manga_recommender_system)")
st.markdown("Model from [Scikit-Surprise](https://surprise.readthedocs.io/en/stable/)")

st.markdown("---")

# Initialize session state if not already
if "username" not in st.session_state:
    st.session_state.username = ""
if "genres_filter" not in st.session_state:
    st.session_state.genres_filter = []
if "tags_filter" not in st.session_state:
    st.session_state.tags_filter = []
if "status_filter" not in st.session_state:
    st.session_state.status_filter = "All"
if "recommendations" not in st.session_state:
    st.session_state.recommendations = pd.DataFrame()

# User login input
st.session_state.username = st.text_input(
    "Enter your username:", st.session_state.username
)

# Fetch recommendations when the button is clicked
if st.button("Get Recommendations"):
    if st.session_state.username.strip() == "":
        st.warning("Please enter a valid username.")
    else:
        # Fetch recommendations only when the username is provided
        st.session_state.recommendations = get_recommendations(
            username=st.session_state.username
        )


# Filter selection (persistent across reruns)
if not st.session_state.recommendations.empty:
    # Dynamically generate genre and tag options from the recommendations DataFrame
    genres_options = (
        st.session_state.recommendations["genres"]
        .str.split(",")
        .explode()
        .str.strip()
        .unique()
        .tolist()
    )
    tags_options = (
        st.session_state.recommendations["tags"]
        .str.split(",")
        .explode()
        .str.strip()
        .unique()
        .tolist()
    )
    status_options = ["All", "RELEASING", "FINISHED", "CANCELLED"]

    # Filters for recommendations
    st.session_state.genres_filter = st.multiselect(
        "Select Genres", options=genres_options, default=st.session_state.genres_filter
    )
    st.session_state.tags_filter = st.multiselect(
        "Select Tags", options=tags_options, default=st.session_state.tags_filter
    )
    st.session_state.status_filter = st.selectbox(
        "Select Status",
        options=status_options,
        index=status_options.index(st.session_state.status_filter),
    )

    # Apply filters and display recommendations
    filtered_recommendations = st.session_state.recommendations.copy()

    # Apply genre filter
    if st.session_state.genres_filter:
        filtered_recommendations = filtered_recommendations[
            filtered_recommendations["genres"].apply(
                lambda x: (
                    any(
                        genre in x.split(", ")
                        for genre in st.session_state.genres_filter
                    )
                    if pd.notnull(x)
                    else True
                )
            )
        ]

    # Apply tag filter
    if st.session_state.tags_filter:
        filtered_recommendations = filtered_recommendations[
            filtered_recommendations["tags"].apply(
                lambda x: (
                    any(tag in x.split(", ") for tag in st.session_state.tags_filter)
                    if pd.notnull(x)
                    else True
                )
            )
        ]

    # Apply status filter
    if st.session_state.status_filter != "All":
        filtered_recommendations = filtered_recommendations[
            filtered_recommendations["status"] == st.session_state.status_filter
        ]

    # Check if there are any recommendations after filtering
    if not filtered_recommendations.empty:
        st.subheader("Here are your top manga recommendations:")
        # Limit to Top 5 recommendations
        filtered_recommendations = filtered_recommendations.head(5)

        for index, row in filtered_recommendations.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 3])

                with col1:
                    st.image(row["cover_image"], width=120, caption=row["title_romaji"])

                with col2:
                    st.markdown(f"**{row['title_romaji']}**")
                    st.markdown(f"*Genre:* {row['genres']}")
                    st.markdown(f"*Predicted Rating:* {row['predicted_rating']}")
                    st.markdown(f"*Mangaka:* {row['mangaka']}")
                    st.markdown(f"*Start Date:* {row['start_date']}")
                    if pd.notnull(row["end_date"]):
                        st.markdown(f"*End Date:* {row['end_date']}")

        st.success("Enjoy your recommended manga!")

    else:
        st.warning("No recommendations found based on your filters.")

else:
    # Check if user exists in the users dataset
    if check_user_exists(st.session_state.username):
        st.info(
            f"Computing recommendations for {st.session_state.username}, "
            "please check a few minutes later!"
        )

        # Trigger Prefect workflow
        success = trigger_recommendation_workflow(st.session_state.username)

        if success:
            st.success(
                "Recommendation generation triggered successfully!"
                " Try to login a few minutes later."
            )
        else:
            st.error(
                "Failed to trigger the recommendation workflow. "
                "We will fix this as soon as possible."
            )
    else:
        st.error("Sorry, username was not found in our users database.")

        # Go fetch user data with anilist api and retrain pipeline
        # Give a message to check again the next day or something
