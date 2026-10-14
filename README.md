# Fake News Detection System

A reproducible NLP/ML project that classifies article text as **REAL** or **FAKE** using TF-IDF features with Multinomial Naive Bayes and tuned Logistic Regression.

> **Important:** This is a text-classification demonstration, not a fact-checking service. A prediction is not proof that a claim is true or false. Model performance depends on the dataset and may not generalize to current events or unseen sources.

## Features

- Text preprocessing and TF-IDF unigram/bigram features
- Multinomial Naive Bayes baseline
- Logistic Regression tuned with cross-validation
- Exact duplicate article text is removed within each class before splitting
- Stratified held-out evaluation with accuracy, precision, recall, F1, confusion matrix, and classification report
- Saved model artifacts and machine-readable metrics
- Command-line prediction helper
- Automated tests for dataset validation, model training, artifact creation, and prediction

## Requirements

- Python 3.10 or newer recommended
- Dataset files `Fake.csv` and `True.csv` (not included in this repository)

## Quick start

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS/Linux**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Follow [DATASET.md](DATASET.md) to obtain and place the two CSV files, then train:

```bash
python fake_news_detection.py train
```

This creates local artifacts in `outputs/` and a reproducible `outputs/metrics.json` report. Metrics are calculated on a stratified holdout split; they are not hard-coded claims and should be regenerated for your dataset.

## Predict

```bash
python fake_news_detection.py predict "Paste the full news article text here"
```

Choose a model explicitly if desired:

```bash
python fake_news_detection.py predict --model naive_bayes "Paste the full news article text here"
python predict_news.py "Paste the full news article text here"
python quick_test.py
```

Train first before running prediction commands. The trained `.joblib` files are deliberately excluded from Git and must not be loaded from untrusted sources.

## Tests

Install dependencies and run:

```bash
python -m unittest discover -s tests -v
```

The tests cover dataset validation, undersized classes, a synthetic end-to-end training/prediction smoke test, and prediction input handling. A full benchmark training run still requires the local datasets and can take several minutes.

## Project structure

```text
fake-news-detection/
├── fake_news_detection.py       # canonical training, evaluation and prediction pipeline
├── predict_news.py               # standalone CLI prediction helper
├── quick_test.py                 # smoke test using trained model artifacts
├── tests/
│   └── test_pipeline.py           # automated unit tests
├── DATASET.md                    # dataset setup and caveats
├── requirements.txt
├── model_requirements.txt
├── MODEL_README.md
├── DEPLOYMENT_GUIDE.md
└── outputs/                       # generated locally; ignored by Git
```

## Labels and limitations

- `FAKE` = label 1 from `Fake.csv`.
- `REAL` = label 0 from `True.csv`.
- Dataset provenance, near-duplicates, publisher/topic leakage, and time drift can inflate offline metrics. Exact duplicate article texts are removed within each class before the holdout split, but this does not eliminate near-duplicate or source leakage. Do not describe the result as verified fact-checking.
- Live scraping is intentionally not mixed into the holdout evaluation. If added later, scraped labels need an independent verification process.

## License

See [LICENSE](LICENSE).


### Dataset size requirement

The training pipeline requires at least **4 non-empty articles in each class** (fake and real) to support the stratified holdout split and 3-fold cross-validation. Keep `Fake.csv` and `True.csv` local; they are not committed to this public repository.
