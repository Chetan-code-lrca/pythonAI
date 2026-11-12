"""Train and use a small, deterministic scikit-learn classifier on Iris."""

from functools import lru_cache
from math import isfinite
from numbers import Real
from typing import Iterable

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COUNT = 4


def _make_classifier(random_state: int) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, random_state=random_state),
    )


def evaluate_iris(random_state: int = 42) -> float:
    """Return hold-out accuracy for a deterministic Iris classification run."""
    if not isinstance(random_state, int):
        raise TypeError("random_state must be an integer")

    dataset = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.25,
        random_state=random_state,
        stratify=dataset.target,
    )
    model = _make_classifier(random_state)
    model.fit(x_train, y_train)
    return float(accuracy_score(y_test, model.predict(x_test)))


@lru_cache(maxsize=4)
def _fitted_iris_classifier(random_state: int) -> tuple[Pipeline, tuple[str, ...]]:
    """Fit once per seed for interactive inference; keep the estimator private."""
    dataset = load_iris()
    model = _make_classifier(random_state)
    model.fit(dataset.data, dataset.target)
    return model, tuple(str(name) for name in dataset.target_names)


def _normalize_samples(
    samples: Iterable[Real] | Iterable[Iterable[Real]],
) -> list[list[float]]:
    """Validate one feature vector or a batch of four-feature vectors."""
    try:
        values = list(samples)
    except TypeError as exc:
        raise ValueError("samples must contain one or more four-feature rows") from exc

    if not values:
        raise ValueError("at least one sample is required")

    if all(isinstance(value, Real) and not isinstance(value, bool) for value in values):
        rows = [values]
    else:
        rows = []
        for row_index, row in enumerate(values, start=1):
            if isinstance(row, (str, bytes)):
                raise ValueError(f"sample {row_index} must be a sequence of numeric features")
            try:
                rows.append(list(row))
            except TypeError as exc:
                raise ValueError(
                    f"sample {row_index} must be a sequence of numeric features"
                ) from exc

    normalized: list[list[float]] = []
    for row_index, row in enumerate(rows, start=1):
        if len(row) != FEATURE_COUNT:
            raise ValueError(
                f"sample {row_index} must contain exactly {FEATURE_COUNT} features"
            )

        normalized_row: list[float] = []
        for feature_index, value in enumerate(row, start=1):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(
                    f"sample {row_index}, feature {feature_index} must be a real number"
                )
            number = float(value)
            if not isfinite(number):
                raise ValueError(
                    f"sample {row_index}, feature {feature_index} must be finite"
                )
            normalized_row.append(number)
        normalized.append(normalized_row)

    return normalized


def predict_iris(
    samples: Iterable[Real] | Iterable[Iterable[Real]],
    random_state: int = 42,
) -> list[str]:
    """Predict Iris species for one four-feature row or a batch of rows.

    Features must be ordered as sepal length, sepal width, petal length, and
    petal width (centimetres). The inference model is fitted on the complete
    bundled Iris dataset and cached by random_state. This convenience API is
    for demonstrating inference; use evaluate_iris() for hold-out evaluation.
    """
    if not isinstance(random_state, int):
        raise TypeError("random_state must be an integer")

    normalized = _normalize_samples(samples)
    model, target_names = _fitted_iris_classifier(random_state)
    predictions = model.predict(normalized)
    return [target_names[int(class_id)] for class_id in predictions]


if __name__ == "__main__":
    print(f"Iris hold-out accuracy: {evaluate_iris():.3f}")
