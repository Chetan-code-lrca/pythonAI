# Python AI Playground

A small, runnable collection of Python machine-learning examples. The first example trains and evaluates a deterministic Iris classifier using scikit-learn; it needs no API key or downloaded dataset.

## Run

```bash
python -m pip install -r requirements.txt
python -m python_ai.classifier
python -m pytest -q
```

## Use the classifier for inference

The reusable API accepts one four-feature observation or a batch. Features must be ordered as sepal length, sepal width, petal length, and petal width (centimetres).

```python
from python_ai.classifier import predict_iris

species = predict_iris([5.1, 3.5, 1.4, 0.2])
print(species)  # ['setosa']

batch = [
    [5.1, 3.5, 1.4, 0.2],
    [6.0, 2.9, 4.5, 1.5],
]
print(predict_iris(batch))
```

The inference model is fitted on the complete bundled Iris dataset and cached by random seed. This convenience API demonstrates prediction; use `evaluate_iris()` for a separate stratified hold-out accuracy estimate. Invalid feature counts, non-numeric values, booleans, NaN, and infinity are rejected before reaching scikit-learn.

## Project rules

- Prefer small, reproducible examples over framework-heavy abstractions.
- Keep datasets out of the repository unless redistribution is permitted.
- Add a regression test whenever behavior or input validation changes.
- Report measured metrics from the actual run; do not promise a fixed accuracy across arbitrary datasets.

## Status

The Iris example is the starter project. Additional examples should be added only when they teach a distinct technique.
