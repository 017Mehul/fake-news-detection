"""Export the trained TF-IDF + Logistic Regression pipeline for browser inference.

Run after training:
    python export_browser_model.py
This writes web/model/model.json. The JSON contains model parameters, not the dataset.
"""
import json
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "logistic_regression_model_grid.joblib"
OUTPUT_PATH = ROOT / "web" / "model" / "model.json"


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. First run: python fake_news_detection.py train"
        )

    pipeline = joblib.load(MODEL_PATH)
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    if len(classifier.classes_) != 2 or list(classifier.classes_) != [0, 1]:
        raise ValueError("Expected binary labels 0=REAL and 1=FAKE.")
    payload = {
        "format_version": 1,
        "model_type": "tfidf_logistic_regression",
        "label_map": {"0": "REAL", "1": "FAKE"},
        "vocabulary": vectorizer.vocabulary_,
        "idf": vectorizer.idf_.tolist(),
        "stop_words": sorted(vectorizer.get_stop_words() or []),
        "token_pattern": r"(?u)\b\w\w+\b",
        "lowercase": bool(vectorizer.lowercase),
        "ngram_range": list(vectorizer.ngram_range),
        "sublinear_tf": bool(vectorizer.sublinear_tf),
        "norm": vectorizer.norm,
        "coefficient": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "classes": [int(value) for value in classifier.classes_],
        "source_model": "logistic_regression_model_grid.joblib",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Exported browser model to {OUTPUT_PATH}")
    print(f"Vocabulary size: {len(payload['vocabulary']):,}")
    print(f"Model JSON size: {OUTPUT_PATH.stat().st_size / (1024 * 1024):.2f} MiB")


if __name__ == "__main__":
    main()
