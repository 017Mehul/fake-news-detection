# Optional FastAPI Deployment

The project has two separate inference paths:

- **Browser app (recommended for the current frontend):** loads `web/model/model.json` and predicts locally. It does not call this API. See [BROWSER_MODEL_SETUP.md](BROWSER_MODEL_SETUP.md).
- **Optional FastAPI service:** `app.py` exposes `/health` and `/predict` and uses a private scikit-learn `.joblib` pipeline. Use this only if you intentionally want a separate API deployment.

## 1. Train the Python model

Place `Fake.csv` and `True.csv` in the repository root (they are intentionally not committed), then run:

```bash
python -m pip install -r requirements.txt
python fake_news_detection.py train
```

This creates model artifacts under `outputs/`. Keep these files private and do not commit them to the public repository.

## 2. Run the API locally

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

- Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

The health endpoint returns `model_missing` if no trained `.joblib` artifact is present under `outputs/`.

## 3. Deploy the API separately

`render.yaml` is a Render Blueprint starting point. The service command is:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

The deployment must have a trusted copy of the trained `outputs/*.joblib` artifact at the path expected by the application. Use a private artifact store or your host’s private file mechanism; do not make the dataset or model artifacts public just to make deployment work.

Set `ALLOWED_ORIGINS` to the exact frontend origin(s), comma-separated. The default wildcard is for convenience and should be narrowed for a production API.

## API contract

Send a JSON request to `POST /predict`:

```json
{ "text": "Full news article text", "model": "logistic_regression" }
```

The response includes the prediction (`REAL` or `FAKE`), model name, confidence, class probabilities, and a disclaimer. The request accepts up to 5,000 characters. A `503` response indicates the trained model artifact is missing.

## Important

The static frontend currently uses browser-side inference and does not use `FAKE_NEWS_API_URL`. Do not change the browser configuration to point at this API unless you also deliberately update the frontend code to use the API contract. Predictions are text-classifier outputs, not independent verification of factual claims.