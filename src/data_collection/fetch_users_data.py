import time

import pandas as pd
import requests
from loguru import logger

# AniList GraphQL Endpoint
API_URL = "https://graphql.anilist.co"

# GraphQL Query to Fetch User Manga List & Statistics
QUERY = """
    query ($username: String) {
    User(name: $username) {
      id
      statistics {
        manga {
          count
          chaptersRead
          volumesRead
          meanScore
          standardDeviation
          genres {
            genre
            count
          }
          releaseYears {
            chaptersRead
            count
            meanScore
          }
          tags {
            tag { name }
            chaptersRead
            meanScore
            count
          }
          staff {
            staff { name { full } }
            chaptersRead
            count
            meanScore
          }
          lengths {
            chaptersRead
            count
            meanScore
          }
        }
      }
    }
    MediaListCollection(userName: $username, type: MANGA) {
      lists {
        name
        entries {
          media {
            id
            title {
              english
            }
            genres
            format
          }
          score
          progress
          status
          priority
          customLists
          startedAt {
            year
          }
          completedAt {
            year
          }
          repeat
          private
          userId
        }
      }
    }
  }
"""


def fetch_user_data(username: str) -> tuple:
    """
    Fetches a user's manga statistics and manga list from AniList API.

    This function retrieves:
    1. **User Statistics** (Total manga read, favorite genres, tags, etc.)
    2. **User Manga List** (Individual manga entries, scores, progress, etc.)

    Args:
        username (str): AniList username.

    Returns:
        tuple: (user_statistics (dict), user_manga_list (list of dicts))
    """
    variables = {"username": username}
    response = requests.post(API_URL, json={"query": QUERY, "variables": variables})

    if response.status_code != 200:
        logger.error(
            f"Error fetching data for {username}: {response.status_code}, "
            f"{response.text}"
        )

        return {}, []

    data = response.json()
    user_data = data["data"]["User"]
    manga_stats = user_data["statistics"]["manga"] if user_data["statistics"] else {}

    # Create User Statistics Dictionary
    user_statistics = {
        "Username": username,
        "Total Manga Read": manga_stats.get("count", None),
        "Total Chapters Read": manga_stats.get("chaptersRead", None),
        "Total Volumes Read": manga_stats.get("volumesRead", None),
        "Mean Score Given": manga_stats.get("meanScore", None),
        "Score Standard Deviation": manga_stats.get("standardDeviation", None),
        "Favorite Genres": (
            ", ".join(
                f"{g['genre']} ({g['count']})" for g in manga_stats.get("genres", [])
            )
        ),
        "Tag Preferences": (
            ", ".join(
                f"{t['tag']['name']} ({t['count']} manga, "
                f"{t['meanScore']} avg score, {t['chaptersRead']} chapters)"
                for t in manga_stats.get("tags", [])
            )
            if "tags" in manga_stats
            else None
        ),
        "Staff Preferences": (
            ", ".join(
                f"{s['staff']['name']['full']} ({s['count']} manga, "
                f"{s['meanScore']} avg score, {s['chaptersRead']} chapters)"
                for s in manga_stats.get("staff", [])
            )
            if "staff" in manga_stats
            else None
        ),
    }

    # Create User Manga List
    user_manga_list = []
    for list_data in data["data"]["MediaListCollection"]["lists"]:
        for entry in list_data["entries"]:
            user_manga_list.append(
                {
                    "Username": username,
                    "Manga Title": entry["media"]["title"]["english"],
                    "Genres": (
                        ", ".join(entry["media"]["genres"])
                        if entry["media"]["genres"]
                        else None
                    ),
                    "Format": entry["media"]["format"],
                    "Score": entry["score"],
                    "Progress": entry["progress"],
                    "Status": entry["status"],
                    "Custom Lists": (
                        ", ".join(entry["customLists"])
                        if entry["customLists"]
                        else None
                    ),
                    "Started Reading": (
                        entry["startedAt"]["year"] if entry["startedAt"] else None
                    ),
                    "Completed Reading": (
                        entry["completedAt"]["year"] if entry["completedAt"] else None
                    ),
                    "Reread Count": entry["repeat"],
                }
            )

    if user_statistics:
        logger.info(f"Fetched statistics for {username}: {user_statistics}")
    if user_manga_list:
        logger.info(
            f"Fetched manga list for {username} - " f"First Entry: {user_manga_list[0]}"
        )
    else:
        logger.warning(f"No manga data found for {username}")

    return user_statistics, user_manga_list


if __name__ == "__main__":
    # Fetch user data and save to CSV
    users_df = pd.read_csv("./data/processed/usernames.csv")
    usernames = users_df.iloc[:, 0].tolist()
    total_users = len(usernames)

    all_user_stats = []
    all_user_manga = []

    for index, user in enumerate(usernames, start=1):
        remaining_users = total_users - index  # Calculate remaining users
        logger.info(
            f"Fetching data for {user} ({index}/{total_users}) - "
            f"{remaining_users} users remaining..."
        )

        stats, manga_list = fetch_user_data(user)

        if stats:
            all_user_stats.append(stats)
        all_user_manga.extend(manga_list)

        time.sleep(2)  # Prevent API rate-limiting

    # Convert to DataFrames
    df_stats = pd.DataFrame(all_user_stats)
    df_manga = pd.DataFrame(all_user_manga)

    logger.debug(f"1st User: {df_stats.head()}")
    logger.debug(f"1st User: {df_manga.head()}")
    logger.info(f"Users Stats Shape: {df_stats.shape}")
    logger.info(f"Users Manga Shape: {df_manga.shape}")

    # Save to CSVs
    df_stats.to_csv("./data/processed/user_statistics_v2.csv", index=False)
    df_manga.to_csv("./data/processed/user_manga_list_v2.csv", index=False)

    logger.info("Users datasets saved successfully!")
