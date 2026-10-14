# Fake News Detection

An NLP and machine-learning project that classifies news article text as **REAL** or **FAKE** using TF-IDF features and Logistic Regression, with a Multinomial Naive Bayes baseline. It includes a Python training/evaluation workflow, command-line prediction, automated tests, and a static browser app that runs an exported model locally.

> **Important:** This is a text classifier, not a fact-checking service. A prediction is not proof that a claim is true or false. Performance on a historical dataset may not generalize to current events, new publishers, satire, or unseen topics.

## Project at a glance

- **NLP:** text cleaning and TF-IDF unigram/bigram features
- **Models:** Logistic Regression and Multinomial Naive Bayes
- **Evaluation:** stratified holdout metrics, precision, recall, F1, confusion matrix, and classification report
- **Browser app:** static frontend with TF-IDF + Logistic Regression inference in the browser
- **CLI:** train, evaluate, and predict from the terminal
- **Tests:** automated Python unit tests and GitHub Actions workflow

## Home Dashboard

![Fake News Detection home dashboard](docs/images/home-dashboard.jpg)

## Try the browser app

The static frontend is in [`web/`](web/). It loads the exported model from [`web/model/model.json`](web/model/model.json) and does not require a prediction API.

### Run locally

From the repository root:

```bash
python -m http.server 8000
```

Then open [http://localhost:8000/web/](http://localhost:8000/web/). Use a local server rather than opening the HTML file directly, so the browser can fetch the model JSON.

For static hosting, deploy the `web/` directory (for example, set the hosting root directory to `web` and use no build command). The configured model path is relative: `./model/model.json`.

See [Browser Model Setup](BROWSER_MODEL_SETUP.md) and the [web app README](web/README.md) for details.

## Run the Python pipeline

### Requirements

- Python 3.10 or newer
- The dataset files `Fake.csv` and `True.csv`, which are not included in this repository

### 1. Create an environment and install dependencies

```bash
python -m venv .venv
```

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**macOS/Linux**

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Prepare the dataset

Follow [DATASET.md](DATASET.md) to obtain and place `Fake.csv` and `True.csv` in the repository root. These dataset files are intentionally excluded from Git.

### 3. Train and evaluate

```bash
python fake_news_detection.py train
```

The command writes trained artifacts and evaluation metrics under `outputs/`. Metrics are generated from the current dataset and training run; do not treat them as fixed project-wide guarantees.

### 4. Predict an article

```bash
python fake_news_detection.py predict "Paste the full news article text here"
```

Select a model explicitly if desired:

```bash
python fake_news_detection.py predict --model logistic_regression "Article text"
python fake_news_detection.py predict --model naive_bayes "Article text"
```

The standalone helper is also available:

```bash
python predict_news.py "Article text"
```

Train the model first. The Python prediction commands need the generated local model artifacts.

## Export or refresh the browser model

The checked-in browser model is an exported TF-IDF + Logistic Regression representation. To regenerate it from a matching trained pipeline:

```bash
python fake_news_detection.py train
python export_browser_model.py
```

Review the exported file before committing it. The exporter expects a compatible trained Logistic Regression pipeline in `outputs/`; do not combine a classifier and vectorizer from different training runs. The browser model JSON contains vocabulary and model weights, so it is public and inspectable. Never commit `Fake.csv`, `True.csv`, or the Python `.joblib` artifacts.

## Tests

Run the automated unit tests:

```bash
python -m unittest discover -s tests -v
```

Optional model smoke tests require locally trained artifacts:

```bash
python quick_test.py
python test_model.py
```

The hand-written sample articles are sanity checks, not a representative benchmark. Use the held-out evaluation generated during training for model assessment. GitHub Actions runs the unit tests on supported Python versions.

## Repository layout

```text
.
├── fake_news_detection.py       # training, evaluation, and CLI prediction
├── predict_news.py              # standalone prediction helper
├── app.py                       # optional FastAPI prediction API
├── export_browser_model.py      # exports the browser inference model
├── tests/test_pipeline.py       # automated unit tests
├── test_model.py                # optional labeled sample checks
├── quick_test.py                # small model smoke test
├── DATASET.md                   # dataset setup and caveats
├── BROWSER_MODEL_SETUP.md       # browser model export and deployment
├── MODEL_README.md              # model artifact guide
├── requirements.txt
└── web/                         # static frontend and exported model
```

The FastAPI service in `app.py` is an optional separate deployment path; the current browser frontend does not call it. API deployment requires private trained `outputs/*.joblib` artifacts and separate configuration. See [API setup](API_SETUP.md) before using that path.

## Labels and limitations

- `FAKE` corresponds to label `1` from `Fake.csv`.
- `REAL` corresponds to label `0` from `True.csv`.
- Removing exact duplicate article text does not eliminate near-duplicate leakage, publisher/topic leakage, or temporal drift.
- Dataset provenance and permitted use should be checked before redistribution or deployment.
- Model confidence is a classifier probability estimate, not evidence that an article has been independently verified.
- Do not use predictions as the sole basis for consequential decisions.

## License

See [LICENSE](LICENSE).