from surprise import (
    NMF,
    SVD,
    BaselineOnly,
    CoClustering,
    KNNBaseline,
    KNNBasic,
    KNNWithMeans,
    KNNWithZScore,
    NormalPredictor,
    SlopeOne,
)

MATRIX_FACTORIZATION_MODELS = {"SVD": SVD(), "NMF": NMF(), "Slope One": SlopeOne()}

KNN_MODELS = {
    "KNN Baseline": KNNBaseline(),
    "KNN Basic": KNNBasic(),
    "KNN with Means": KNNWithMeans(),
    "KNN with Z-Score": KNNWithZScore(),
}

BASELINE_MODELS = {
    "Normal Predictor": NormalPredictor(),
    "Baseline Only": BaselineOnly(),
    "Co-clustering": CoClustering(),
}
