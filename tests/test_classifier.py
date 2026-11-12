import pytest

from python_ai.classifier import evaluate_iris, predict_iris


def test_iris_classifier_reaches_reasonable_holdout_accuracy():
    assert evaluate_iris() >= 0.85


def test_iris_evaluation_is_reproducible():
    assert evaluate_iris() == evaluate_iris()


def test_random_state_must_be_an_integer():
    with pytest.raises(TypeError, match="random_state must be an integer"):
        evaluate_iris("42")


def test_predict_iris_returns_species_names_for_a_batch():
    samples = [
        [5.1, 3.5, 1.4, 0.2],
        [6.0, 2.9, 4.5, 1.5],
        [6.5, 3.0, 5.2, 2.0],
    ]

    assert predict_iris(samples) == ["setosa", "versicolor", "virginica"]


def test_predict_iris_accepts_a_single_feature_vector():
    assert predict_iris([5.1, 3.5, 1.4, 0.2]) == ["setosa"]


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        ([], "at least one sample is required"),
        ([[5.1, 3.5, 1.4]], "sample 1 must contain exactly 4 features"),
        ([[5.1, "wide", 1.4, 0.2]], "sample 1, feature 2 must be a real number"),
        ([[5.1, float("nan"), 1.4, 0.2]], "sample 1, feature 2 must be finite"),
        ([[5.1, float("inf"), 1.4, 0.2]], "sample 1, feature 2 must be finite"),
        ([True, 3.5, 1.4, 0.2], "sample 1, feature 1 must be a real number"),
    ],
)
def test_predict_iris_rejects_invalid_samples(samples, message):
    with pytest.raises(ValueError, match=message):
        predict_iris(samples)


def test_predict_iris_rejects_non_integer_random_state():
    with pytest.raises(TypeError, match="random_state must be an integer"):
        predict_iris([[5.1, 3.5, 1.4, 0.2]], random_state="42")
