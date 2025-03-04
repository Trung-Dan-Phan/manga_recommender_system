SELECT
    uml.*,
    usr.`Score Format`,
    mcr.*
FROM `manga-recommender-system.users_dataset.user_manga_list_raw` AS uml
JOIN `manga-recommender-system.users_dataset.user_statistics_raw` AS usr
    ON uml.`Username` = usr.`Username`
JOIN `manga-recommender-system.manga_dataset.manga_collection_raw` AS mcr
    ON uml.`Title Romaji` = mcr.`Title Romaji`
WHERE uml.`Score` > 0
AND uml.`Title Romaji` IS NOT NULL;
