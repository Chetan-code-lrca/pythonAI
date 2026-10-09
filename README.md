# Python AI Playground

A small, runnable collection of Python machine-learning examples. The first example trains and evaluates a deterministic Iris classifier using scikit-learn; it needs no API key or downloaded dataset.

## Run

```bash
python -m pip install -r requirements.txt
python -m python_ai.classifier
python -m pytest -q
```

## Project rules

- Prefer small, reproducible examples over framework-heavy abstractions.
- Keep datasets out of the repository unless redistribution is permitted.
- Add a regression test whenever behavior or input validation changes.
- Report measured metrics from the actual run; do not promise a fixed accuracy across arbitrary datasets.

## Status

The Iris example is the starter project. Additional examples should be added only when they teach a distinct technique.
