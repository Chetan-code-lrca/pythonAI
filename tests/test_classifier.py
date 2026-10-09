from python_ai.classifier import evaluate_iris


def test_iris_classifier_reaches_reasonable_holdout_accuracy():
    assert evaluate_iris() >= 0.85


def test_iris_evaluation_is_reproducible():
    assert evaluate_iris() == evaluate_iris()


def test_random_state_must_be_an_integer():
    try:
        evaluate_iris("42")
    except TypeError as exc:
        assert str(exc) == "random_state must be an integer"
    else:
        raise AssertionError("Expected TypeError for non-integer random_state")
