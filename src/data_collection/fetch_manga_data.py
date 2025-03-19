import time

import pandas as pd
import requests
from loguru import logger

from utils.bigquery_utils import write_data_to_bigquery
from utils.fetching_utils import get_last_fetched_page

# AniList GraphQL Endpoint
API_URL = "https://graphql.anilist.co"

# GraphQL Query
QUERY = """
query ($page: Int) {
  Page(page: $page, perPage: 100) {
    media(type: MANGA) {
      id
      title {
        romaji
        english
      }
      genres
      description
      averageScore
      popularity
      status
      chapters
      volumes
      coverImage {
        large
      }
      tags {
        name
      }
      startDate {
        year
        month
        day
      }
      endDate {
        year
        month
        day
      }
      isAdult
      format
      rankings {
        rank
        type
        format
      }
      relations {
        edges {
          relationType
          node {
            id
            title {
              romaji
              english
            }
            type
            status
          }
        }
      }
      staff {
        edges {
          node {
            id
            name {
              full
            }
            primaryOccupations
          }
          role
        }
      }
      reviews {
        nodes {
          score
          summary
          rating
        }
      }
    }
  }
}
"""


def fetch_manga_data_utils(
    start_page: int = 1, max_pages: int = 50, max_manga: int = 2000
) -> list:
    """
    Fetch new manga data from AniList API, ensuring no duplicates.
    Resumes from the last fetched page instead of restarting at page 1.

    Args:
        start_page (int): The page to start fetching from.
        max_pages (int): Maximum number of pages to fetch.
        max_manga (int): The maximum number of manga to fetch in one run.

    Returns:
        list: A list of new manga entries.
    """
    page = start_page
    all_manga = []
    fetched_ids = set()  # Track manga IDs to avoid duplicates

    while len(all_manga) < max_manga and page < start_page + max_pages:
        variables = {"page": page}
        response = requests.post(API_URL, json={"query": QUERY, "variables": variables})

        if response.status_code != 200:
            logger.error(f"Error: {response.status_code}, {response.text}")
            break

        data = response.json()
        manga_list = data["data"]["Page"]["media"]

        if not manga_list:
            logger.info(f"No manga found at page {page}. Stopping...")
            break  # Stop when no more data

        for manga in manga_list:
            manga_id = manga["id"]

            # Skip duplicate manga
            if manga_id in fetched_ids:
                continue  # Avoid duplicates

            fetched_ids.add(manga_id)

            # Filter staff to include only "Mangaka"
            mangaka_staff = [
                staff["node"]["name"]["full"]
                for staff in manga["staff"]["edges"]
                if "Mangaka" in staff["node"]["primaryOccupations"]
            ]

            all_manga.append(
                {
                    "ID": manga["id"],
                    "Title Romaji": manga["title"]["romaji"],
                    "Title English": manga["title"]["english"],
                    "Genres": ", ".join(manga["genres"]) if manga["genres"] else None,
                    "Description": manga["description"],
                    "Average Score": manga["averageScore"],
                    "Popularity": manga["popularity"],
                    "Status": manga["status"],
                    "Chapters": manga["chapters"],
                    "Volumes": manga["volumes"],
                    "Cover Image": (
                        manga["coverImage"]["large"] if manga["coverImage"] else None
                    ),
                    "Tags": (
                        ", ".join(tag["name"] for tag in manga["tags"])
                        if manga["tags"]
                        else None
                    ),
                    "Start Date": (
                        f"{manga['startDate']['year']}"
                        if manga["startDate"]["year"]
                        else None
                    ),
                    "End Date": (
                        f"{manga['endDate']['year']}"
                        if manga["endDate"]["year"]
                        else None
                    ),
                    "Is Adult": manga["isAdult"],
                    "Ranking": (
                        ", ".join(
                            f"{rank['rank']} ({rank['type']} - {rank['format']})"
                            for rank in manga["rankings"]
                        )
                        if manga["rankings"]
                        else None
                    ),
                    "Relations": (
                        ", ".join(
                            f"{relation['node']['title']['romaji']} ({relation['relationType']})"
                            for relation in manga["relations"]["edges"]
                            if relation["node"]["title"]["romaji"]
                        )
                        if manga["relations"]["edges"]
                        else None
                    ),
                    "Mangaka": ", ".join(mangaka_staff) if mangaka_staff else None,
                    "Reviews": (
                        ", ".join(
                            f"{review['score']} - {review['summary']}"
                            for review in manga["reviews"]["nodes"]
                        )
                        if manga["reviews"]["nodes"]
                        else None
                    ),
                    "Page": page,
                }
            )

            # Stop if we've reached the max limit
            if len(all_manga) >= max_manga:
                logger.info(f"Reached max manga fetch limit ({max_manga}). Stopping...")
                break

        logger.info(f"Fetched page {page} with {len(manga_list)} manga entries.")
        page += 1
        time.sleep(2)  # Prevent API rate-limiting

    return all_manga


if __name__ == "__main__":
    dataset_id = "manga_dataset"
    table_id = "manga_collection_raw_test"

    # Get last fetched manga ID
    last_fetched_page = get_last_fetched_page(dataset_id=dataset_id, table_id=table_id)
    logger.info(f"Last fetched page: {last_fetched_page}")

    # Fetch new manga data with a max limit of 2000
    new_manga_data = fetch_manga_data_utils(start_page=last_fetched_page + 1)

    if new_manga_data:
        new_manga_df = pd.DataFrame(new_manga_data)

        # Append to BigQuery
        write_data_to_bigquery(
            dataset_id=dataset_id, table_id=table_id, df=new_manga_df, mode="append"
        )

        logger.info(f"Saved {len(new_manga_df)} new manga records!")

    else:
        logger.info("No new manga found.")
