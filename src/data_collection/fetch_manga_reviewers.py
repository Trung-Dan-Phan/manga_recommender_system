import requests
import time
import pandas as pd
from loguru import logger

# AniList API URL
API_URL = "https://graphql.anilist.co"

# Query to fetch popular manga
QUERY_POPULAR_MANGA = """
query ($page: Int) {
  Page(page: $page, perPage: 50) {
    media(type: MANGA, sort: POPULARITY_DESC) {
      id
      title {
        english
      }
    }
  }
}
"""

# Query to fetch reviewers for a manga
QUERY_REVIEWERS = """
query ($mediaId: Int, $page: Int) {
  Media(id: $mediaId) {
    reviews(page: $page, perPage: 50, sort: RATING_DESC) {
      nodes {
        user {
          id
          name
        }
      }
    }
  }
}
"""

def fetch_popular_manga(pages=3):
    """
    Fetches popular manga to use their IDs for fetching reviews.
    
    Args:
        pages (int): Number of pages to fetch (each page = 50 manga).
    
    Returns:
        list: A list of popular manga IDs.
    """
    manga_ids = []
    for page in range(1, pages + 1):
        response = requests.post(API_URL, json={"query": QUERY_POPULAR_MANGA, "variables": {"page": page}})
        if response.status_code != 200:
            logger.error(f"Failed to fetch manga list: {response.status_code}")
            continue

        data = response.json()
        manga_list = data["data"]["Page"]["media"]
        for manga in manga_list:
            manga_ids.append((manga["id"], manga["title"]["english"]))

        logger.info(f"Fetched {len(manga_list)} manga from page {page}.")
        time.sleep(1)  # Avoid hitting rate limits

    return manga_ids

def fetch_reviewers_for_manga(manga_id):
    """
    Fetches reviewers for a given manga ID.
    
    Args:
        manga_id (int): The AniList manga ID.
    
    Returns:
        list: A list of usernames who reviewed the manga.
    """
    reviewers = set()  # Use a set to avoid duplicates
    page = 1

    while True:
        response = requests.post(API_URL, json={"query": QUERY_REVIEWERS, "variables": {"mediaId": manga_id, "page": page}})
        if response.status_code != 200:
            logger.error(f"Failed to fetch reviewers for manga {manga_id}: {response.status_code}")
            break

        data = response.json()
        review_nodes = data["data"]["Media"]["reviews"]["nodes"]

        if not review_nodes:
            break  # No more reviews to fetch

        for review in review_nodes:
            reviewers.add(review["user"]["name"])

        logger.info(f"Fetched {len(review_nodes)} reviewers from manga {manga_id}, page {page}.")
        page += 1
        time.sleep(1)  # Avoid rate limits

    return list(reviewers)

def fetch_all_reviewers():
    """
    Fetches reviewers from multiple popular mangas.
    
    Returns:
        list: A list of unique reviewers from all mangas.
    """
    manga_ids = fetch_popular_manga(pages=3)  # Fetch 150 popular mangas
    all_reviewers = set()  # Store unique usernames

    for manga_id, title in manga_ids:
        logger.info(f"Fetching reviewers for manga: {title} (ID: {manga_id})")
        reviewers = fetch_reviewers_for_manga(manga_id)
        all_reviewers.update(reviewers)

        logger.info(f"Total unique reviewers so far: {len(all_reviewers)}")
        time.sleep(2)  # Avoid rate limits

    return list(all_reviewers)


if __name__ == "__main__":
    # Run the function to get a large set of reviewers
    reviewer_list = fetch_all_reviewers()

    # Save to CSV
    df_reviewers = pd.DataFrame(reviewer_list, columns=["Username"])
    df_reviewers.to_csv("./data/raw/anilist_users_v4.csv", index=False)

    logger.info(f"Saved {len(reviewer_list)} unique reviewers")
