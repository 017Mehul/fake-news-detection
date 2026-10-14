# Dataset setup

The training script expects two CSV files in this directory:

- `Fake.csv`: fake-news articles; must contain a `text` column.
- `True.csv`: real-news articles; must contain a `text` column.

Download the dataset from a source whose license and terms permit your intended use. If using the widely circulated Kaggle "Fake and real news dataset", verify the current dataset page, author attribution, and license before redistributing it. Do not commit the CSV files to this public repository; `.gitignore` excludes them.

After downloading, extract the files and place both CSV files beside `fake_news_detection.py`. The expected directory structure is:

```text
fake-news-detection/
├── Fake.csv                 # local only, not committed
├── True.csv                 # local only, not committed
├── fake_news_detection.py
└── requirements.txt
```

The pipeline uses the `text` column only, strips blank entries, and removes exact duplicate article texts within each class before creating the train/test split. Near-duplicates and source/topic leakage may still affect evaluation. It does not append live scraped articles to the evaluation dataset, because scraped labels are not independently verified ground truth.
