from collections import defaultdict

from loguru import logger


def precision_recall_at_k(predictions: list, k: int = 10, threshold: int = 7):
    """
    Compute Precision@K and Recall@K for recommendations.

    Precision@K: Measures the proportion of relevant items in the top-K recommendations.
    Recall@K: Measures the proportion of relevant items that were successfully recommended.

    Parameters:
    - predictions (list): List of tuples from Surprise `test()` method.
    - k (int): Number of top recommendations to consider.
    - threshold (float): The minimum rating to consider an item relevant.

    Returns:
    - precisions (dict): Precision@K scores per user.
    - recalls (dict): Recall@K scores per user.
    """
    # Step 1: Map predictions to each user
    user_est_true = defaultdict(list)

    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls = {}, {}

    # Step 2: Compute Precision@K and Recall@K for each user
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated rating (descending order)
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Compute relevant counts
        n_rel = sum(
            (true_r >= threshold) for (_, true_r) in user_ratings
        )  # Relevant items
        n_rec_k = sum(
            (est >= threshold) for (est, _) in user_ratings[:k]
        )  # Recommended items
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # Compute Precision@K (Avoid division by zero)
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Compute Recall@K (Avoid division by zero)
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    # Logging summary statistics
    avg_precision = sum(precisions.values()) / len(precisions) if precisions else 0
    avg_recall = sum(recalls.values()) / len(recalls) if recalls else 0

    logger.debug(
        f"Average Precision@{k}: {avg_precision:.4f}, Average Recall@{k}: {avg_recall:.4f}"
    )

    return precisions, recalls
