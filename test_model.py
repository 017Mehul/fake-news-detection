"""Smoke-test the trained fake-news pipelines on labeled example articles.

These hand-written examples are sanity checks, NOT a statistically valid benchmark.
Use `python fake_news_detection.py train` and its held-out metrics for evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
MODEL_PATHS = {
    "naive_bayes": OUTPUTS / "naive_bayes_model.joblib",
    "logistic_regression": OUTPUTS / "logistic_regression_model_grid.joblib",
}

# Expected labels are illustrative; a model's output does not establish truth.
TEST_ARTICLES = [
    {"id": "F01", "expected": 1, "text": "NASA has secretly admitted that Earth is flat and all satellite images were fabricated to protect its budget."},
    {"id": "R01", "expected": 0, "text": "The central bank said its policy committee voted to raise the benchmark interest rate by 0.25 percentage points in its published statement."},
    {"id": "F02", "expected": 1, "text": "BREAKING: Hillary Clinton arrested in a secret military operation, while a deep state roundup is being hidden by every mainstream outlet."},
    {"id": "R02", "expected": 0, "text": "Apple reported quarterly revenue in its earnings release and said sales of its products increased in several international markets."},
    {"id": "F03", "expected": 1, "text": "Leaked documents prove drinking bleach cures cancer and COVID-19, but pharmaceutical companies are suppressing this miracle cure."},
    {"id": "R03", "expected": 0, "text": "The Supreme Court issued a 6-3 ruling in a voting rights case. The opinion and docket entry are available on the court's official website."},
    {"id": "F04", "expected": 1, "text": "Government officials finally confessed that 5G towers spread coronavirus by weakening people's immune systems after a long cover-up."},
    {"id": "R04", "expected": 0, "text": "The Bureau of Labor Statistics released its monthly employment report, stating that the unemployment rate was 3.7 percent."},
    {"id": "F05", "expected": 1, "text": "A miracle herb guarantees instant weight loss for everyone, doctors hate this secret, and no clinical testing is needed."},
    {"id": "R05", "expected": 0, "text": "The city council approved its annual budget after a public meeting. The agenda, vote record, and budget document were posted online."},
    {"id": "F06", "expected": 1, "text": "Scientists confirm that the moon is an artificial hologram and world leaders have ordered the media to hide the evidence."},
    {"id": "R06", "expected": 0, "text": "The university announced the dates for its admissions process and published application requirements on the official admissions website."},
    {"id": "F07", "expected": 1, "text": "Forward this message to ten people or your bank account will be frozen tomorrow, according to an anonymous insider."},
    {"id": "R07", "expected": 0, "text": "The company filing says the board approved the transaction on Tuesday. The full filing is available in the regulator's public database."},
    {"id": "F08", "expected": 1, "text": "A secret government memo proves that every vaccine contains a tracking chip, but the document has no verifiable source."},
    {"id": "R08", "expected": 0, "text": "Researchers published their methods and findings in a peer-reviewed journal and noted that further studies are needed to replicate the results."},
]


def load_pipeline(name: str):
    path = MODEL_PATHS[name]
    if not path.is_file():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    model = joblib.load(path)
    if hasattr(model, "named_steps"):
        if not {"tfidf", "classifier"}.issubset(model.named_steps):
            raise TypeError(f"{path} pipeline must contain tfidf and classifier steps.")
        return model
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise TypeError(f"{path} is not a probability-capable classifier or pipeline.")
    vectorizer_path = OUTPUTS / "tfidf_vectorizer.joblib"
    if not vectorizer_path.is_file():
        raise FileNotFoundError(
            f"Standalone classifier detected, but {vectorizer_path} is missing. "
            "Place the matching tfidf_vectorizer.joblib in outputs/."
        )
    vectorizer = joblib.load(vectorizer_path)
    if not hasattr(vectorizer, "transform"):
        raise TypeError(f"{vectorizer_path} is not a fitted vectorizer.")
    if getattr(model, "n_features_in_", None) != len(vectorizer.vocabulary_):
        raise ValueError("Classifier/vectorizer feature-count mismatch; use matching artifacts from the same training run.")
    return (vectorizer, model)


def evaluate_model(name: str) -> dict[str, Any]:
    loaded = load_pipeline(name)
    texts = [item["text"] for item in TEST_ARTICLES]
    y_true = np.array([item["expected"] for item in TEST_ARTICLES], dtype=int)
    if isinstance(loaded, tuple):
        vectorizer, classifier = loaded
        X = vectorizer.transform(texts)
        predictions = np.asarray(classifier.predict(X), dtype=int)
        probabilities = np.asarray(classifier.predict_proba(X), dtype=float)
        classes = list(classifier.classes_)
    else:
        predictions = np.asarray(loaded.predict(texts), dtype=int)
        probabilities = np.asarray(loaded.predict_proba(texts), dtype=float)
        classes = list(loaded.classes_)
    if classes != [0, 1] or probabilities.shape != (len(texts), 2):
        raise ValueError(f"{name} must return probability columns ordered [REAL=0, FAKE=1].")
    rows = []
    for article, expected, predicted, prob in zip(TEST_ARTICLES, y_true, predictions, probabilities):
        rows.append({
            "id": article["id"], "expected": "FAKE" if expected == 1 else "REAL",
            "predicted": "FAKE" if predicted == 1 else "REAL",
            "correct": bool(expected == predicted), "real_probability": float(prob[0]),
            "fake_probability": float(prob[1]), "text": article["text"],
        })
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision_fake": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "f1_fake": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "confusion_matrix_labels_real_fake": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_true, predictions, labels=[0, 1],
            target_names=["REAL", "FAKE"], zero_division=0, output_dict=True),
        "samples": len(TEST_ARTICLES),
        "warning": "Hand-written smoke-test examples only; not a reliable estimate of real-world generalization.",
        "predictions": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test trained fake-news model pipelines.")
    parser.add_argument("--model", choices=["naive_bayes", "logistic_regression", "both"], default="both")
    parser.add_argument("--json-out", type=Path, help="Optional path to write detailed results as JSON.")
    args = parser.parse_args()
    names = list(MODEL_PATHS) if args.model == "both" else [args.model]

    results: dict[str, Any] = {
        "purpose": "sanity-check only; not a benchmark",
        "label_map": {"0": "REAL", "1": "FAKE"},
        "models": {},
    }
    for name in names:
        result = evaluate_model(name)
        results["models"][name] = result
        print(f"\n{'=' * 72}\n{name.upper().replace('_', ' ')}\n{'=' * 72}")
        print(f"Accuracy:          {result['accuracy']:.3f}")
        print(f"Balanced accuracy: {result['balanced_accuracy']:.3f}")
        print(f"Fake precision:    {result['precision_fake']:.3f}")
        print(f"Fake recall:       {result['recall_fake']:.3f}")
        print(f"Fake F1:           {result['f1_fake']:.3f}")
        print("Confusion matrix (rows=true [REAL, FAKE], columns=predicted [REAL, FAKE]):")
        print(np.array(result["confusion_matrix_labels_real_fake"]))
        for row in result["predictions"]:
            mark = "PASS" if row["correct"] else "CHECK"
            print(f"{row['id']} {mark:5} expected={row['expected']:4} predicted={row['predicted']:4} "
                  f"p(fake)={row['fake_probability']:.3f}")
    print("\nNOTE: Hand-written examples are sanity checks, not trustworthy benchmark data.")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Detailed JSON written to: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
