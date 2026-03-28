# Fake News Detection System

A machine learning system that detects fake news with **94-99% accuracy** using Natural Language Processing and two classification algorithms.

## 🎯 Features

- **Two ML Models**: Multinomial Naive Bayes (94.36%) & Logistic Regression (98.91%)
- **TF-IDF Vectorization**: Converts text to numerical features
- **Trained on 44,898 articles**: Professional-grade dataset
- **Interactive Testing**: Test any article in real-time
- **Production Ready**: Complete with model persistence

## 📊 Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 94.36% | 94.75% | 94.44% | 94.59% |
| Logistic Regression | 98.91% | 99.21% | 98.70% | 98.95% |

**Note**: Naive Bayes is recommended for real-world use as it generalizes better to new articles.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn beautifulsoup4 requests joblib
```

### 2. Prepare Data

Download the dataset from Kaggle:
```bash
pip install kaggle
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset
unzip fake-and-real-news-dataset.zip
```

You should have:
- `Fake.csv` (23,481 fake news articles)
- `True.csv` (21,417 real news articles)

### 3. Train the Model

```bash
python fake_news_detector_complete.py
```

This will:
- Load and preprocess the data
- Train both models
- Evaluate performance
- Save models to `model_outputs/`

### 4. Test the Model

```bash
python fake_news_detector_complete.py predict
```

Or run training and then test interactively.

## 💻 Usage Examples

### Training

```python
from fake_news_detector_complete import main

# Train models
vectorizer, nb_model, lr_model = main()
```

### Prediction

```python
from fake_news_detector_complete import load_models, predict_news

# Load trained models
vectorizer, nb_model, lr_model = load_models()

# Test an article
article = "Scientists confirm Earth is flat..."
result, conf_real, conf_fake = predict_news(article, vectorizer, nb_model)

print(f"Prediction: {result}")
print(f"Confidence: {conf_fake:.2%}")
```

### Interactive Mode

```python
from fake_news_detector_complete import interactive_mode

interactive_mode()
```

## 📁 File Structure

```
.
├── fake_news_detector_complete.py  # Main system file
├── Fake.csv                         # Fake news dataset
├── True.csv                         # Real news dataset
├── model_outputs/                   # Saved models
│   ├── vectorizer.joblib
│   ├── naive_bayes.joblib
│   └── logistic_regression.joblib
└── MODEL_README.md                  # This file
```

## 🔧 Configuration

Edit these parameters in the code:

```python
# TF-IDF Parameters
TFIDF_PARAMS = {
    'stop_words': 'english',
    'max_df': 0.8,
    'min_df': 5,
    'ngram_range': (1, 2),
    'max_features': 10000
}

# Train/Test Split
test_size = 0.20  # 80/20 split
```

## 🧪 Testing Examples

### Fake News Examples:
- "Scientists confirm Earth is flat and NASA has been lying for decades"
- "Hillary Clinton arrested in secret military operation"
- "5G towers spreading coronavirus by weakening immune systems"

### Real News Examples:
- "Federal Reserve raises interest rates by 0.25 percentage points"
- "Apple reports quarterly earnings exceeding analyst expectations"
- "Supreme Court rules in 6-3 decision on voting rights case"

## 📈 Model Details

### Naive Bayes
- **Algorithm**: Multinomial Naive Bayes
- **Best for**: Generalization to new articles
- **Accuracy**: 94.36%
- **Strengths**: Fast, reliable, handles unseen data well

### Logistic Regression
- **Algorithm**: L2-regularized Logistic Regression
- **Best for**: High accuracy on similar data
- **Accuracy**: 98.91%
- **Hyperparameters**: C=1.0, solver=liblinear

## 🎓 How It Works

1. **Text Preprocessing**: Clean and standardize text data
2. **Feature Extraction**: Convert text to TF-IDF vectors (10,000 features)
3. **Model Training**: Train Naive Bayes and Logistic Regression
4. **Prediction**: Classify new articles as REAL or FAKE
5. **Confidence Scores**: Provide probability estimates

## ⚠️ Limitations

- Works best on political news (dataset bias)
- May struggle with satire or opinion pieces
- Requires retraining for different domains
- Performance depends on article length and quality

## 🔮 Future Improvements

- [ ] Add deep learning models (BERT, RoBERTa)
- [ ] Expand to multiple languages
- [ ] Include source credibility scoring
- [ ] Add real-time web scraping
- [ ] Create web API/interface

## 📝 License

This project is for educational purposes. Dataset from Kaggle (CC-BY-NC-SA-4.0).

## 🤝 Contributing

Feel free to improve the model, add features, or fix bugs!

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Built with ❤️ using Python and scikit-learn**
