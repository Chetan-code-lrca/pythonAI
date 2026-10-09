"""Train a small, deterministic scikit-learn classifier on Iris."""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, random_state=random_state),
    )
    model.fit(x_train, y_train)
    return float(accuracy_score(y_test, model.predict(x_test)))


if __name__ == "__main__":
    print(f"Iris hold-out accuracy: {evaluate_iris():.3f}")
