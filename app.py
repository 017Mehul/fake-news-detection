import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fake_news_detection import predict_news

app = FastAPI(title="Fake News Detection API", version="1.0.0")
origins_setting = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = ["*"] if origins_setting.strip() == "*" else [item.strip() for item in origins_setting.split(",") if item.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=False, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"])

class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    model: str = "logistic_regression"

@app.get("/")
def root():
    return {"service": "Fake News Detection API", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    from pathlib import Path
    outputs = Path(__file__).resolve().parent / "outputs"
    return {"status": "ok" if any(outputs.glob("*.joblib")) else "model_missing"}

@app.post("/predict")
def predict(request: PredictionRequest):
    article = request.text.strip()
    if not article:
        raise HTTPException(status_code=422, detail="Please enter a news article.")
    if request.model not in ("logistic_regression", "naive_bayes"):
        raise HTTPException(status_code=422, detail="model must be logistic_regression or naive_bayes")
    try:
        result = predict_news(article, request.model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Trained model missing. Train locally and provide the private outputs/*.joblib artifact to this service.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    confidence = result.get("confidence")
    probabilities = result.get("probabilities", {})
    return {
        "prediction": result["label"],
        "model": result["model"],
        "confidence": confidence,
        "confidence_percent": round(confidence * 100, 2) if confidence is not None else None,
        "real_probability": probabilities.get("real"),
        "fake_probability": probabilities.get("fake"),
        "disclaimer": "Machine-learning classification only; this is not independent fact verification."
    }
