# Model and Artifact Guide

The canonical pipeline is `fake_news_detection.py`.

## Training

1. Install dependencies: `pip install -r requirements.txt`.
2. Place `Fake.csv` and `True.csv` in the project root as described in [DATASET.md](DATASET.md).
3. Run `python fake_news_detection.py train`.

## Generated artifacts

The training command creates `outputs/` locally:

- `naive_bayes_model.joblib`
- `logistic_regression_model_grid.joblib`
- `metrics.json`

The TF-IDF vectorizer is bundled inside each scikit-learn Pipeline, so preprocessing and inference stay paired with the model. Artifacts are excluded from Git by design. Do not load joblib files from untrusted sources.

## Evaluation

Metrics are written from a stratified held-out split to `outputs/metrics.json`. Do not quote a fixed accuracy figure unless you have run the current code against a documented dataset version and retained the output. High accuracy on this benchmark does not prove the system can fact-check current news.

## Prediction

```bash
python fake_news_detection.py predict "Full article text"
python fake_news_detection.py predict --model naive_bayes "Full article text"
```
