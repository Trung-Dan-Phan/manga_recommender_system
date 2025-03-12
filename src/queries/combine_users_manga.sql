SELECT *
FROM `manga-recommender-system.recommendations_dataset.recommendations` AS recs
JOIN `manga-recommender-system.manga_dataset.manga_collection_clean` AS manga
USING (`title_romaji`)
ORDER BY `predicted_rating` DESC
