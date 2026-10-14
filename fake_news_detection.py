"""Reproducible fake-news detection training and evaluation pipeline.

Dataset expected in the repository root:
  Fake.csv  (fake articles)
  True.csv  (real articles)

Artifacts are generated locally in outputs/ and are intentionally not committed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
FAKE_CSV = ROOT / "Fake.csv"
TRUE_CSV = ROOT / "True.csv"
NAIVE_BAYES_PATH = OUT_DIR / "naive_bayes_model.joblib"
LOGISTIC_PATH = OUT_DIR / "logistic_regression_model_grid.joblib"
METRICS_PATH = OUT_DIR / "metrics.json"
RANDOM_STATE = 42


def load_dataset(fake_path: Path = FAKE_CSV, true_path: Path = TRUE_CSV) -> Tuple[pd.Series, pd.Series]:
    """Load the standard Fake.csv/True.csv files and return text and labels."""
    if not fake_path.is_file():
        raise FileNotFoundError(f"Missing dataset: {fake_path}. See DATASET.md.")
    if not true_path.is_file():
        raise FileNotFoundError(f"Missing dataset: {true_path}. See DATASET.md.")

    fake = pd.read_csv(fake_path)
    real = pd.read_csv(true_path)
    for name, frame in (("Fake.csv", fake), ("True.csv", real)):
        if "text" not in frame.columns:
            raise ValueError(f"{name} must contain a 'text' column.")
    fake_text = fake["text"].fillna("").astype(str).str.strip()
    real_text = real["text"].fillna("").astype(str).str.strip()
    # Remove exact duplicate article text before splitting to prevent copies from
    # appearing in both training and holdout sets and inflating evaluation scores.
    fake_text = fake_text[fake_text.ne("")].drop_duplicates().reset_index(drop=True)
    real_text = real_text[real_text.ne("")].drop_duplicates().reset_index(drop=True)
    X = pd.concat([fake_text, real_text], ignore_index=True)
    y = pd.Series([1] * len(fake_text) + [0] * len(real_text), name="label")
    class_counts = y.value_counts()
    if y.nunique() != 2 or len(y) < 8:
        raise ValueError("Dataset must contain at least 8 unique non-empty articles across both classes.")
    if int(class_counts.min()) < 4:
        raise ValueError(
            "Each class must contain at least 4 unique non-empty articles for the stratified holdout split and 3-fold cross-validation."
        )
    return X, y


def build_models():
    """Return models whose vectorizers are fitted only on training text."""
    nb = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.95, min_df=2, ngram_range=(1, 2))),
        ("classifier", MultinomialNB()),
    ])
    lr = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.95, min_df=2, ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1500, random_state=RANDOM_STATE, class_weight="balanced")),
    ])
    return nb, lr


def train_and_evaluate() -> dict:
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    nb, lr = build_models()
    nb.fit(X_train, y_train)
    search = GridSearchCV(
        lr,
        param_grid={"classifier__C": [0.1, 1.0, 10.0]},
        cv=3,
        scoring="f1_macro",
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    models = {"naive_bayes": nb, "logistic_regression": search.best_estimator_}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["naive_bayes"], NAIVE_BAYES_PATH)
    joblib.dump(models["logistic_regression"], LOGISTIC_PATH)
    metrics = {
        "dataset": {"samples": int(len(y)), "fake_samples": int((y == 1).sum()), "real_samples": int((y == 0).sum()), "fake_label": 1, "real_label": 0,
                    "train_samples": int(len(y_train)), "test_samples": int(len(y_test)),
                    "random_state": RANDOM_STATE, "test_size": 0.2},
        "models": {},
        "note": "Metrics are for the fixed stratified holdout split; dataset source/quality can affect generalization."
    }
    for name, model in models.items():
        pred = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="binary", zero_division=0
        )
        metrics["models"][name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "classification_report": classification_report(
                y_test, pred, target_names=["Real", "Fake"], zero_division=0, output_dict=True
            ),
        }
    metrics["models"]["logistic_regression"]["best_params"] = search.best_params_
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved model artifacts to {OUT_DIR}")
    print(f"Saved evaluation metrics to {METRICS_PATH}")
    for name, result in metrics["models"].items():
        print(f"{name}: accuracy={result['accuracy']:.4f}, precision={result['precision']:.4f}, "
              f"recall={result['recall']:.4f}, F1={result['f1']:.4f}")
    return metrics


def load_model(model_name: str = "logistic_regression"):
    paths = {"naive_bayes": NAIVE_BAYES_PATH, "logistic_regression": LOGISTIC_PATH}
    if model_name not in paths:
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {', '.join(paths)}")
    path = paths[model_name]
    if not path.is_file():
        raise FileNotFoundError(f"Model not found at {path}. Run 'python fake_news_detection.py train' first.")
    return joblib.load(path)


def predict_news(text: str, model_name: str = "logistic_regression") -> dict:
    """Return a prediction and model probability when supported; fake=1, real=0."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Please provide non-empty news text.")
    model = load_model(model_name)
    prediction = int(model.predict([text])[0])
    result = {"label": "FAKE" if prediction == 1 else "REAL", "prediction": prediction, "model": model_name}
    classifier = model.named_steps.get("classifier") if hasattr(model, "named_steps") else model
    if hasattr(classifier, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        result["confidence"] = float(max(probabilities))
        result["probabilities"] = {"real": float(probabilities[0]), "fake": float(probabilities[1])}
    return result


def main():
    parser = argparse.ArgumentParser(description="Train and use a TF-IDF fake-news text classifier.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="Train models and write held-out metrics.")
    predict_parser = sub.add_parser("predict", help="Classify a news article using saved artifacts.")
    predict_parser.add_argument("text", help="News article text in quotes.")
    predict_parser.add_argument("--model", choices=["naive_bayes", "logistic_regression"], default="logistic_regression")
    args = parser.parse_args()
    if args.command == "train":
        train_and_evaluate()
    else:
        print(json.dumps(predict_news(args.text, args.model), indent=2))


if __name__ == "__main__":
    main()
