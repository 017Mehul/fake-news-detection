# Connecting the web UI to the trained model

The frontend calls `POST /predict` on the FastAPI service. The model and vectorizer stay together in the saved scikit-learn Pipeline; do not upload the dataset or model artifacts to the public Git repository.

## 1. Train the model locally

Place your own `Fake.csv` and `True.csv` in the repository root (these files are intentionally not committed), then run:

```bash
pip install -r requirements.txt
python fake_news_detection.py train
```

This creates these local model files:

- `outputs/logistic_regression_model_grid.joblib`
- `outputs/naive_bayes_model.joblib`

## 2. Run the API locally

From the repository root:

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/health`. The selected model must show as available. Open `http://127.0.0.1:8000/docs` to test `POST /predict`.

## 3. Point the web UI at the API

The web page loads `web/config.js`. It currently points to `http://127.0.0.1:8000` for local development. After deploying the API, change `window.FAKE_NEWS_API_URL` to the API's HTTPS base URL, commit that change, and redeploy the static frontend. Do not add a trailing slash.

## 4. Deploy the API separately

The repository includes `render.yaml` as a Render Blueprint starting point. The API service needs Python dependencies from `requirements.txt` and starts with:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

**Important:** the trained `outputs/*.joblib` files are git-ignored and are not included in the deployment automatically. Provide the trained model artifact through your hosting provider's private file/secret-file mechanism at the exact `outputs/` path expected by the app, or use a private artifact download step. Never make the trained artifacts or dataset public just to make deployment work. Check `/health` after adding the model file; a `model_missing` status means the API is running but cannot predict yet.

Set `ALLOWED_ORIGINS` on the API service to the exact frontend origin(s) for production, comma-separated. The default wildcard is convenient for local testing only.

## API contract

Request:

```json
{"text": "Full news article text", "model": "logistic_regression"}
```

Response includes `prediction` (`REAL` or `FAKE`), `confidence_percent`, and class probabilities. A 503 response means the trained model artifact is missing. Predictions are text-classifier outputs, not independent verification of factual claims.
