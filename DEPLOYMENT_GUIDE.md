# Deployment Guide

## Current project scope

This repository currently provides a Python command-line training, evaluation, and prediction workflow. It does **not** include a hosted API or browser frontend. For local usage, follow the [README](README.md).

## Local setup

1. Install Python 3.10+.
2. Create and activate a virtual environment.
3. Run `pip install -r requirements.txt`.
4. Obtain the datasets and place `Fake.csv` and `True.csv` in the repository root; see [DATASET.md](DATASET.md).
5. Run `python fake_news_detection.py train`.
6. Test a prediction with `python fake_news_detection.py predict "Example article text"`.

## Hosting a web service

To deploy a web API, add an API layer (for example FastAPI) with a startup check for the generated model artifacts, request validation, input size limits, and error handling. Train the model in a controlled build/release workflow or provide artifacts through a trusted artifact store. Do not commit private datasets or large trained artifacts to a public repository by default.

A frontend is a separate optional component and should call the API rather than duplicate model inference in the browser. Neither the API nor frontend is implemented in the current repository.

## Important limitations

- Predictions are model classifications, not verified fact-checks.
- Dataset terms and provenance must be checked before deployment.
- Evaluate temporal drift, publisher/topic leakage, and false-positive/false-negative costs before real-world use.
- Do not accept arbitrary serialized model uploads from users.
