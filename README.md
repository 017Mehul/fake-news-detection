# 📰 Fake News Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Yes-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

🚀 A high-performance machine learning system that classifies news as **REAL or FAKE** using NLP and ML techniques.

🔍 Achieves:
- 99%+ accuracy (Logistic Regression)
- 95%+ accuracy (Naive Bayes)

💡 Built with: Python, scikit-learn, TF-IDF, NLP

📌 Ideal for:
- Detecting misinformation
- NLP learning projects
# 📰 Fake News Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Yes-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

🚀 A high-performance machine learning system that classifies news as **REAL or FAKE** using NLP and ML techniques.

🔍 Achieves:
- 99%+ accuracy (Logistic Regression)
- 95%+ accuracy (Naive Bayes)

💡 Built with: Python, scikit-learn, TF-IDF, NLP

📌 Ideal for:
- Detecting misinformation
- NLP learning projects
- ML portfolio showcase

## 📸 Demo

![Demo Screenshot](link_here)

## ⚡ Quick Start

```bash
git clone https://github.com/017Mehul/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt
python fake_news_detection.py
```

## 🧪 Example Prediction

```text
Text: "Breaking: Government announces new policy"
Prediction: REAL ✅ (Confidence: 97.2%)

Text: "You won't believe this shocking secret..."
Prediction: FAKE ❌ (Confidence: 98.5%)
```

## 🛠 Tech Stack

- Python
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- BeautifulSoup (web scraping)
- TF-IDF (NLP)


## 🎯 Why This Project?

Fake news is a major global issue. This project demonstrates:
- NLP preprocessing techniques
- Feature engineering using TF-IDF
- Model comparison and evaluation
- Real-world ML pipeline design

📖 Full documentation below ⬇️

## Overview

This system implements two classical text classification approaches to detect fake news:
- **Multinomial Naive Bayes (MNB)**: Fast baseline classifier achieving 95%+ accuracy
- **Logistic Regression (LR)**: Optimized with GridSearchCV achieving 99%+ accuracy

The system provides comprehensive evaluation metrics including confusion matrices, ROC curves, precision-recall curves, and learning curves to assess model performance and diagnose overfitting/underfitting.

## Features

- **Data Collection**: Load labeled datasets from CSV files and optionally augment with web-scraped news from BBC, CNN, and Reuters
- **Preprocessing**: Automatic text column detection, missing value handling, and stratified train/test splitting
- **Feature Extraction**: TF-IDF vectorization with bigrams and optimized parameters
- **Model Training**: MNB baseline and LR with hyperparameter tuning via GridSearchCV
- **Evaluation**: Comprehensive metrics (accuracy, precision, recall, F1) with binary, macro, and weighted averaging
- **Visualization**: Interactive plots for confusion matrices, ROC curves, PR curves, learning curves, and error analysis
- **Persistence**: Automatic model and vectorizer saving/loading to avoid retraining

## Requirements

### Python Version
- Python 3.8 or higher

### Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
beautifulsoup4>=4.10.0
requests>=2.26.0
joblib>=1.1.0
```

## Installation

1. Clone or download this repository

2. Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn beautifulsoup4 requests joblib
```

Or using a requirements.txt file (if provided):
```bash
pip install -r requirements.txt
```

3. Prepare your dataset files (see Dataset Format section below)

## Dataset Format

### Required CSV Files

The system expects two CSV files in the same directory as the script:

1. **Fake.csv**: Contains fake news articles with label=1
2. **True.csv**: Contains real news articles with label=0

### CSV Column Requirements

Each CSV file must contain at least one of the following text columns:
- `text` (preferred)
- `title`
- `headline`
- `content`

If none of these standard column names exist, the system will use the first string/object column as the text column.

### Example CSV Structure

```csv
title,text,subject,date
"Breaking: Major Event Happens","Full article text here...",politics,2023-01-15
"Another Headline","More article content...",worldnews,2023-01-16
```

**Note**: The system automatically detects and uses the appropriate text column, so you don't need to rename columns manually.

## Usage

### Basic Usage

Simply run the script:

```bash
python fake_news_detection.py
```

The system will:
1. Load data from Fake.csv and True.csv
2. Optionally scrape additional real news from BBC, CNN, and Reuters
3. Preprocess and split the data (80% train, 20% test)
4. Extract TF-IDF features
5. Train or load both MNB and LR models
6. Evaluate models and display metrics
7. Generate interactive visualizations

### Output Files

All outputs are saved to the `outputs/` directory:

```
outputs/
├── updated_news_dataset.csv              # Combined dataset
├── tfidf_vectorizer.joblib               # Fitted TF-IDF vectorizer
├── naive_bayes_model.joblib              # Trained MNB model
├── logistic_regression_model_grid.joblib # Trained LR model
├── MultinomialNaiveBayes_classification_report.txt
└── LogisticRegression_classification_report.txt
```

### Model Persistence

The system automatically saves trained models and the vectorizer. On subsequent runs:
- If model files exist, they are loaded instead of retraining
- This saves time and ensures consistent predictions
- To retrain models, simply delete the `.joblib` files in the `outputs/` directory

## Configuration

You can modify the configuration constants at the top of `fake_news_detection.py`:

```python
# Random state for reproducibility
RANDOM_STATE = 42

# Output directory
OUT_DIR = "outputs"

# TF-IDF parameters
TFIDF_PARAMS = {
    'stop_words': 'english',      # Remove common English words
    'max_df': 0.7,                # Ignore terms in >70% of documents
    'ngram_range': (1, 2),        # Use unigrams and bigrams
    'max_features': 30000         # Limit vocabulary size
}

# CSV file paths
FAKE_CSV = "Fake.csv"
TRUE_CSV = "True.csv"

# Logistic Regression hyperparameter grid
GRID_PARAMS = {
    'C': [0.01, 0.1, 1.0, 5.0],   # Regularization strength
    'penalty': ['l2'],             # L2 regularization
    'solver': ['liblinear', 'lbfgs']  # Optimization algorithms
}
```

## Architecture

### Pipeline Flow

```
Data Collection → Preprocessing → Feature Extraction → Model Training → Evaluation → Visualization
```

### Module Structure

1. **Configuration Module**: Centralized constants and parameters
2. **Data Collection Module**: CSV loading and web scraping
3. **Preprocessing Module**: Text cleaning, column detection, train/test splitting
4. **Feature Extraction Module**: TF-IDF vectorization
5. **Model Training Module**: MNB and LR training with hyperparameter tuning
6. **Evaluation Module**: Metrics calculation and classification reports
7. **Visualization Module**: Plot generation for model analysis
8. **Persistence Module**: Model and data saving/loading

## Evaluation Metrics

The system calculates comprehensive metrics for both models:

### Accuracy
Overall correctness of predictions

### Binary Metrics (Positive Class = Fake News)
- **Precision**: Of all predicted fake news, how many were actually fake?
- **Recall**: Of all actual fake news, how many did we detect?
- **F1-Score**: Harmonic mean of precision and recall

### Macro-Averaged Metrics
Treats both classes (real and fake) equally, regardless of class distribution

### Weighted-Averaged Metrics
Accounts for class imbalance by weighting metrics by support

### Confusion Matrix
Shows true positives, true negatives, false positives, and false negatives

### ROC Curve and AUC
Receiver Operating Characteristic curve showing classifier performance across all thresholds

### Precision-Recall Curve
Shows the tradeoff between precision and recall at different thresholds

### Learning Curves
Diagnose overfitting (large gap between train and validation) or underfitting (high error)

## Visualizations

The system generates interactive plots using matplotlib:

1. **Confusion Matrix**: Visual representation of prediction accuracy
2. **ROC Curve**: Shows true positive rate vs false positive rate with AUC score
3. **Precision-Recall Curve**: Shows precision vs recall tradeoff
4. **Accuracy Comparison**: Bar chart comparing both models
5. **Learning Curves**: Training and validation accuracy vs training set size
6. **Error Curves**: Training and validation error vs training set size

All plots are displayed interactively using `plt.show()` and can be saved manually.

## Web Scraping

The system optionally scrapes real news headlines from credible sources:
- **BBC News**: https://www.bbc.com/news
- **CNN**: https://www.cnn.com
- **Reuters**: https://www.reuters.com

### Scraping Features
- Up to 20 headlines per source
- 8-second timeout to prevent hanging
- Graceful error handling (continues with CSV data if scraping fails)
- All scraped headlines labeled as real news (label=0)

### Disabling Web Scraping

If you want to skip web scraping, you can comment out the scraping call in the main execution:

```python
# scraped_df = get_all_real_news()
scraped_df = pd.DataFrame(columns=['text', 'label'])  # Empty DataFrame
```

## Error Handling

The system handles various error scenarios gracefully:

### File Not Found
If CSV files are missing, the system raises a descriptive error:
```
FileNotFoundError: Fake news CSV file not found: Fake.csv
```

### Network Errors
If web scraping fails, the system prints a warning and continues with CSV data only

### Data Quality Issues
If no suitable text column is found, the system raises:
```
ValueError: No suitable text column found in DataFrame
```

### Model Training Issues
sklearn exceptions are raised for convergence failures or memory errors

## Performance

### Computational Efficiency
- **MNB Training**: < 5 seconds
- **LR GridSearchCV**: < 2 minutes (with 4-fold CV and parallel processing)
- **Memory Usage**: < 2GB for typical datasets
- **Prediction Latency**: < 1 second for 10,000 samples

### Scalability
- Current implementation handles datasets up to ~1M samples in memory
- For larger datasets, consider using `HashingVectorizer` or incremental learning
- GridSearchCV uses `n_jobs=-1` for parallel processing across all CPU cores

## Troubleshooting

### Issue: "No suitable text column found"
**Solution**: Ensure your CSV has at least one column with text data (string/object type)

### Issue: Models not training
**Solution**: Check if model files already exist in `outputs/` directory. Delete them to force retraining.

### Issue: Web scraping fails
**Solution**: This is normal and expected. The system continues with CSV data only. Check your internet connection if you need scraped data.

### Issue: GridSearchCV takes too long
**Solution**: Reduce the parameter grid in `GRID_PARAMS` or use `RandomizedSearchCV` instead

### Issue: Memory error during training
**Solution**: Reduce `max_features` in `TFIDF_PARAMS` or use a smaller dataset

## Advanced Usage

### Using Trained Models for Prediction

```python
import joblib
import pandas as pd

# Load trained model and vectorizer
model = joblib.load('outputs/logistic_regression_model_grid.joblib')
vectorizer = joblib.load('outputs/tfidf_vectorizer.joblib')

# Prepare new text
new_texts = ["Breaking: Shocking news headline here"]

# Transform and predict
X_new = vectorizer.transform(new_texts)
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)

# Interpret results
for text, pred, prob in zip(new_texts, predictions, probabilities):
    label = "FAKE" if pred == 1 else "REAL"
    confidence = max(prob) * 100
    print(f"Text: {text}")
    print(f"Prediction: {label} (Confidence: {confidence:.2f}%)")
```

### Customizing TF-IDF Parameters

```python
# Experiment with different parameters
TFIDF_PARAMS = {
    'stop_words': 'english',
    'max_df': 0.8,              # More lenient on common terms
    'ngram_range': (1, 3),      # Include trigrams
    'max_features': 50000       # Larger vocabulary
}
```

### Customizing Hyperparameter Grid

```python
# More extensive grid search
GRID_PARAMS = {
    'C': [0.001, 0.01, 0.1, 1.0, 10.0],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}
```

## Future Enhancements

Potential improvements for the system:

1. **Advanced Models**: Implement transformer-based models (BERT, RoBERTa)
2. **Feature Engineering**: Add metadata features (source, author, publish date)
3. **Explainability**: Use LIME or SHAP for model interpretability
4. **Real-time Processing**: Stream processing for live news feeds
5. **Web Interface**: Create a web dashboard or browser extension
6. **Ensemble Methods**: Combine multiple models for better accuracy

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- All functions have docstrings
- New features include appropriate error handling
- Tests are added for new functionality

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the error messages carefully
3. Ensure all dependencies are installed correctly
4. Verify your CSV files match the expected format

## Acknowledgments

This system uses:
- **scikit-learn**: Machine learning algorithms and evaluation metrics
- **pandas**: Data manipulation and analysis
- **BeautifulSoup**: Web scraping functionality
- **matplotlib/seaborn**: Data visualization

## Citation

If you use this system in your research, please cite:

```
Fake News Detection System
A machine learning approach using Multinomial Naive Bayes and Logistic Regression
https://github.com/yourusername/fake-news-detection
```

## 👨‍💻 Author

**Mehul Gupta**

🚀 Aspiring ML Engineer | NLP Enthusiast | App and Web Developer

- 📧 Email: [Add Email here]
- 🔗 LinkedIn: [Add LinkedIn link here]
- 💻 Portfolio: [Add portfolio link here]
