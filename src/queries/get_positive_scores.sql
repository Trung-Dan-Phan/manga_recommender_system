-- This query retrieves all users who rated mangas
-- We remove scores of 0 which could indicate the manga hasn't been rated
SELECT
    uml.*,
    usr.`Score Format`
FROM `manga-recommender-system.users_dataset.user_manga_list_raw` AS uml
JOIN `manga-recommender-system.users_dataset.user_statistics_raw` AS usr
ON uml.`Username` = usr.`Username`
WHERE `Score` > 0
AND `Title Romaji` is not null
;
