import time

import pandas as pd
import requests
from loguru import logger

# AniList API URL
API_URL = "https://graphql.anilist.co"

# GraphQL Query to Fetch Users Updating Their Manga List for a Specific Manga
QUERY_MANGA_ACTIVITY = """
query ($mediaId: Int, $page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    activities(mediaId: $mediaId, type: MEDIA_LIST, sort: ID_DESC) {
      ... on ListActivity {
        user {
          name
        }
      }
    }
  }
}
"""


def fetch_users_who_updated_manga(
    media_id: int, start_page: int, max_pages: int
) -> set:
    """
    Fetches users who updated their manga status for a specific AniList manga.

    Args:
        media_id (int): The AniList media ID of the manga.

    Returns:
        set: Set of unique usernames
    """
    usernames = set()
    page = start_page

    while page <= start_page + max_pages:
        variables = {"mediaId": media_id, "page": page}
        response = requests.post(
            API_URL, json={"query": QUERY_MANGA_ACTIVITY, "variables": variables}
        )

        if response.status_code != 200:
            logger.error(
                f"Failed to fetch activity for media {media_id}, "
                f"page {page}: {response.status_code}"
            )
            break

        data = response.json()
        activities = data["data"]["Page"]["activities"]

        if not activities:
            break  # Stop when no more activities are available

        for activity in activities:
            usernames.add(activity["user"]["name"])

        logger.info(
            f"Fetched {len(activities)} activity updates for media {media_id}, page {page}."
        )
        page += 1
        time.sleep(2)  # Avoid API rate-limits

    return usernames


if __name__ == "__main__":
    # Fetch users who updated their status for specified manga
    media_id = 30013  # One Piece
    users = fetch_users_who_updated_manga(
        media_id=media_id, start_page=1, max_pages=100
    )

    # Save to CSV if data exists
    df_activity = pd.DataFrame(list(users), columns=["Username"])
    df_activity.to_csv("./data/raw/anilist_users_v3.csv", index=False)
    logger.info(f"Saved {len(users)} usernames!")
