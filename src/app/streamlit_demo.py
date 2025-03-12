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
DEPLOYMENT_ID = "6e868437-442b-4bae-be41-67579e2dcc52"

# Prefect API URL (Replace with actual Prefect workflow trigger URL)
PREFECT_WORKFLOW_URL = (
    "http://127.0.0.1:4200/api/deployments/{DEPLOYMENT_ID}/create_flow_run"
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
def trigger_recommendation_workflow(username: str):
    response = requests.post(PREFECT_WORKFLOW_URL, json={"username": username})
    return response.status_code == 200  # Returns True if successful


# Streamlit UI
st.title("Manga Recommender Demo")
st.write("By Trung Dan Phan [GitHub](https://github.com/Trung-Dan-Phan)")
st.markdown("[Source code](https://github.com/Trung-Dan-Phan/manga_recommender_system)")
st.markdown("Model from [Scikit-Surprise](https://surprise.readthedocs.io/en/stable/)")

st.markdown("---")

# User login input
username = st.text_input("Enter your username:")

if st.button("Get Recommendations"):
    if username.strip() == "":
        st.warning("Please enter a valid username.")
    else:
        recommendations = get_recommendations(username=username)

        if not recommendations.empty:
            st.subheader("Here are your top manga recommendations:")

            # Limit to Top 5 recommendations
            recommendations = recommendations.head(5)

            for index, row in recommendations.iterrows():
                with st.container():
                    col1, col2 = st.columns([1, 3])

                    with col1:
                        st.image(
                            row["cover_image"], width=120, caption=row["title_romaji"]
                        )

                    with col2:
                        st.markdown(f"**{row['title_romaji']}**")
                        st.markdown(f"*Genre:* {row['genres']}")
                        st.markdown(f"*Predicted Rating:* {row['predicted_rating']}")

            st.success("Enjoy your recommended manga!")

        else:
            # Check if user exists in the users dataset
            if check_user_exists(username):
                st.info(
                    f"Computing recommendations for {username}, please check a few minutes later!"
                )

                # Trigger Prefect workflow
                success = trigger_recommendation_workflow(username)

                if success:
                    st.success(
                        "Recommendation generation triggered successfully! Try to login again."
                    )
                else:
                    st.error(
                        "Failed to trigger the recommendation workflow. Please try again later."
                    )
            else:
                st.error("Sorry, username was not found in our users database.")
