import time

import pandas as pd
import requests
from loguru import logger

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


def fetch_manga_data(start_page: int, max_pages: int) -> dict:
    """
    Fetch manga data from AniList API and filter for relevant attributes.

    This function retrieves multiple pages of manga data, including details like titles,
    genres, average scores, popularity, and staff details filtered to include only
    primary "Mangaka" (and not Translators).
    The function handles API pagination until no more results are found.

    Returns:
        list: A list of dictionaries, each representing a manga with relevant data.
    """
    page = start_page
    all_manga = []

    while page <= start_page + max_pages:
        variables = {"page": page}
        response = requests.post(API_URL, json={"query": QUERY, "variables": variables})

        if response.status_code != 200:
            logger.error(f"Error: {response.status_code}, {response.text}")
            break

        data = response.json()
        manga_list = data["data"]["Page"]["media"]

        if not manga_list:
            break  # Stop when no more data

        for manga in manga_list:
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
                }
            )

        # Log the first manga entry for debugging
        if page == start_page and all_manga:
            logger.debug(f"First Manga Entry: {all_manga[0]}")

        logger.info(f"Fetched page {page} with {len(manga_list)} manga entries.")
        page += 1
        time.sleep(2)  # Prevent API rate-limiting

    return all_manga


if __name__ == "__main__":
    # Fetch manga data in batches and save to CSV
    manga_data = fetch_manga_data(start_page=1, max_pages=100)
    manga_df = pd.DataFrame(manga_data)

    manga_df.to_csv("./data/raw/manga_data_batch_4.csv", index=False)

    logger.info(f"Saved {len(manga_df)} manga data!")
