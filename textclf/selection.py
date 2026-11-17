"""The one rule that picks the "best" model.

Every tuned model is scored by stratified k-fold cross-validation on the training set (the same
folds for every model, see `cv_folds`). The best model is the one with the highest mean CV
accuracy; ties go to the simpler model, in the order of SIMPLICITY_ORDER. Test accuracy is
reported alongside but never used to choose, so it stays an honest estimate for the chosen model.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold

CV_FOLDS = 5
CV_COLUMN = "CV Accuracy"
# Model keys from simplest to most complex; earlier wins a tie on CV accuracy.
SIMPLICITY_ORDER = ["MNB", "MLR", "KNN", "SVM", "random_forest", "GBM"]


def cv_folds(n_splits=CV_FOLDS, random_state=8):
    """The shared, seeded stratified folds every model is scored on."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def select_best(cv_scores):
    """Return the key with the highest CV accuracy, breaking ties by SIMPLICITY_ORDER.

    `cv_scores` maps model keys to mean cross-validated accuracy on the training set.
    Scores within 1e-9 of each other count as a tie.
    """
    if not cv_scores:
        raise ValueError("no models to choose from")
    top = max(cv_scores.values())
    tied = [key for key, score in cv_scores.items() if np.isclose(score, top, rtol=0, atol=1e-9)]
    rank = {key: i for i, key in enumerate(SIMPLICITY_ORDER)}
    return min(tied, key=lambda key: rank.get(key, len(rank)))
