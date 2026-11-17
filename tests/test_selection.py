"""The model-selection rule: highest CV accuracy on the training set, ties to the simpler model."""
import pytest

from textclf.selection import SIMPLICITY_ORDER, select_best
from textclf import MODELS


def test_highest_cv_accuracy_wins():
    assert select_best({"MNB": 0.91, "KNN": 0.97, "SVM": 0.95}) == "KNN"


def test_tie_goes_to_simpler_model():
    assert select_best({"random_forest": 0.95, "SVM": 0.95, "KNN": 0.93}) == "SVM"
    assert select_best({"GBM": 1.0, "MNB": 1.0, "MLR": 1.0}) == "MNB"


def test_float_noise_counts_as_tie():
    assert select_best({"SVM": 0.9, "MLR": 0.9 + 1e-12}) == "MLR"


def test_training_and_test_accuracy_play_no_part():
    # Only CV scores are passed in: a model that fits the training set perfectly is not penalized.
    assert select_best({"KNN": 0.98, "MNB": 0.93}) == "KNN"


def test_every_model_has_a_simplicity_rank():
    assert set(SIMPLICITY_ORDER) == set(MODELS)


def test_empty_raises():
    with pytest.raises(ValueError):
        select_best({})
