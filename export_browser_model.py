"""Export browser model from a pipeline or separate classifier/vectorizer artifacts."""
import json
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "logistic_regression_model_grid.joblib"
VECTORIZER_PATH = ROOT / "outputs" / "tfidf_vectorizer.joblib"
OUTPUT_PATH = ROOT / "web" / "model" / "model.json"

def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing trained classifier: {MODEL_PATH}")
    saved = joblib.load(MODEL_PATH)
    if hasattr(saved, "named_steps"):
        vectorizer = saved.named_steps.get("tfidf")
        classifier = saved.named_steps.get("classifier")
        if vectorizer is None or classifier is None:
            raise TypeError("Pipeline must contain 'tfidf' and 'classifier' steps.")
        source_vectorizer = None
    else:
        if not VECTORIZER_PATH.is_file():
            raise FileNotFoundError(f"Standalone classifier detected; matching vectorizer missing: {VECTORIZER_PATH}")
        vectorizer = joblib.load(VECTORIZER_PATH)
        classifier = saved
        source_vectorizer = VECTORIZER_PATH.name
    if not hasattr(vectorizer, "vocabulary_") or not hasattr(vectorizer, "idf_"):
        raise TypeError("Expected a fitted TF-IDF vectorizer.")
    if not hasattr(classifier, "coef_") or list(classifier.classes_) != [0, 1]:
        raise TypeError("Expected binary Logistic Regression with labels 0=REAL and 1=FAKE.")
    if classifier.coef_.shape[1] != len(vectorizer.idf_):
        raise ValueError("Classifier/vectorizer feature-count mismatch. Use artifacts from the same training run.")
    payload = {
        "format_version": 1, "model_type": "tfidf_logistic_regression",
        "label_map": {"0": "REAL", "1": "FAKE"}, "vocabulary": vectorizer.vocabulary_,
        "idf": vectorizer.idf_.tolist(), "stop_words": sorted(vectorizer.get_stop_words() or []),
        "token_pattern": r"(?u)\\b\\w\\w+\\b", "lowercase": bool(vectorizer.lowercase),
        "ngram_range": list(vectorizer.ngram_range), "sublinear_tf": bool(vectorizer.sublinear_tf),
        "norm": vectorizer.norm, "coefficient": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]), "classes": [int(v) for v in classifier.classes_],
        "source_model": MODEL_PATH.name, "source_vectorizer": source_vectorizer,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Exported browser model to {OUTPUT_PATH}")
    print(f"Vocabulary size: {len(payload['vocabulary']):,}")
    print(f"Model JSON size: {OUTPUT_PATH.stat().st_size / (1024 * 1024):.2f} MiB")

if __name__ == "__main__":
    main()
