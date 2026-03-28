"""
Fake News Detection System

A machine learning system that classifies news headlines and articles as real or fake
using Multinomial Naive Bayes and Logistic Regression with TF-IDF features.
"""

import os
import warnings
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import requests
import joblib

# Suppress sklearn warnings
warnings.filterwarnings('ignore')

# Configuration Constants
RANDOM_STATE = 42
OUT_DIR = "outputs"
TFIDF_PARAMS = {
    'stop_words': 'english',
    'max_df': 0.7,
    'ngram_range': (1, 2),
    'max_features': 30000
}
FAKE_CSV = "Fake.csv"
TRUE_CSV = "True.csv"
GRID_PARAMS = {
    'C': [0.01, 0.1, 1.0, 5.0],
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [1000]
}

# Create outputs directory
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================================
# Data Collection Module
# ============================================================================

def load_datasets(fake_path: str, true_path: str) -> tuple:
    """
    Load and combine fake and true CSV datasets.
    
    Args:
        fake_path: Path to fake news CSV file
        true_path: Path to true news CSV file
    
    Returns:
        Tuple of (fake_df, true_df) DataFrames with label column added
    
    Raises:
        FileNotFoundError: If either CSV file doesn't exist
    """
    # Validate file existence
    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"Fake news CSV file not found: {fake_path}")
    if not os.path.exists(true_path):
        raise FileNotFoundError(f"True news CSV file not found: {true_path}")
    
    # Load CSV files
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    
    # Assign labels: 1 for fake news, 0 for true news
    fake_df['label'] = 1
    true_df['label'] = 0
    
    print(f"Loaded {len(fake_df)} fake news articles")
    print(f"Loaded {len(true_df)} true news articles")

    return fake_df, true_df
def scrape_bbc() -> pd.DataFrame:
    """
    Scrape up to 20 headlines from BBC News.

    Returns:
        DataFrame with columns ['text', 'label'] containing scraped headlines
        Returns empty DataFrame with correct columns on failure
    """
    try:
        url = "https://www.bbc.com/news"
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        headlines = []
        for tag in soup.find_all(['h2', 'h3'], limit=20):
            text = tag.get_text(strip=True)
            if text and len(text) > 10: 
                headlines.append(text)
        if headlines:
            df = pd.DataFrame({'text': headlines[:20], 'label': 0})
            print(f"Scraped {len(df)} headlines from BBC News")
            return df
        else:
            print("No headlines found on BBC News")
            return pd.DataFrame(columns=['text', 'label'])
            
    except Exception as e:
        print(f"BBC scraping failed: {e}")
        return pd.DataFrame(columns=['text', 'label'])


def scrape_cnn() -> pd.DataFrame:
    """
    Scrape up to 20 headlines from CNN.
    
    Returns:
        DataFrame with columns ['text', 'label'] containing scraped headlines
        Returns empty DataFrame with correct columns on failure
    """
    try:
        url = "https://www.cnn.com"
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        headlines = []
        
        # CNN uses span tags with specific classes for headlines
        for tag in soup.find_all(['span', 'h2', 'h3'], limit=50):
            text = tag.get_text(strip=True)
            if text and len(text) > 10:  # Filter out very short text
                headlines.append(text)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_headlines = []
        for h in headlines:
            if h not in seen:
                seen.add(h)
                unique_headlines.append(h)
                if len(unique_headlines) >= 20:
                    break
        
        if unique_headlines:
            df = pd.DataFrame({'text': unique_headlines[:20], 'label': 0})
            print(f"Scraped {len(df)} headlines from CNN")
            return df
        else:
            print("No headlines found on CNN")
            return pd.DataFrame(columns=['text', 'label'])
            
    except Exception as e:
        print(f"CNN scraping failed: {e}")
        return pd.DataFrame(columns=['text', 'label'])


def scrape_reuters() -> pd.DataFrame:
    """
    Scrape up to 20 headlines from Reuters.
    
    Returns:
        DataFrame with columns ['text', 'label'] containing scraped headlines
        Returns empty DataFrame with correct columns on failure
    """
    try:
        url = "https://www.reuters.com"
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        headlines = []
        
        # Reuters uses various heading and link tags for headlines
        for tag in soup.find_all(['h2', 'h3', 'a'], limit=50):
            text = tag.get_text(strip=True)
            if text and len(text) > 10:  # Filter out very short text
                headlines.append(text)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_headlines = []
        for h in headlines:
            if h not in seen:
                seen.add(h)
                unique_headlines.append(h)
                if len(unique_headlines) >= 20:
                    break
        
        if unique_headlines:
            df = pd.DataFrame({'text': unique_headlines[:20], 'label': 0})
            print(f"Scraped {len(df)} headlines from Reuters")
            return df
        else:
            print("No headlines found on Reuters")
            return pd.DataFrame(columns=['text', 'label'])
            
    except Exception as e:
        print(f"Reuters scraping failed: {e}")
        return pd.DataFrame(columns=['text', 'label'])


def get_all_real_news() -> pd.DataFrame:
    """
    Aggregate all scraped news sources.
    
    Returns:
        Combined DataFrame from all scraped sources with label=0
    """
    print("\nScraping real news from credible sources...")
    
    bbc_df = scrape_bbc()
    cnn_df = scrape_cnn()
    reuters_df = scrape_reuters()
    
    # Combine all scraped data
    scraped_dfs = [bbc_df, cnn_df, reuters_df]
    scraped_dfs = [df for df in scraped_dfs if not df.empty]
    
    if scraped_dfs:
        combined_scraped = pd.concat(scraped_dfs, ignore_index=True)
        print(f"\nTotal scraped headlines: {len(combined_scraped)}")
        return combined_scraped
    else:
        print("\nNo headlines scraped from any source")
        return pd.DataFrame(columns=['text', 'label'])


def aggregate_all_data(fake_df: pd.DataFrame, true_df: pd.DataFrame, 
                       scraped_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate fake, true, and scraped data into single DataFrame.
    
    Args:
        fake_df: DataFrame with fake news (label=1)
        true_df: DataFrame with true news (label=0)
        scraped_df: DataFrame with scraped news (label=0)
    
    Returns:
        Combined DataFrame with all data
    """
    # Combine all datasets
    all_dfs = [fake_df, true_df]
    
    if not scraped_df.empty:
        all_dfs.append(scraped_df)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\nCombined dataset statistics:")
    print(f"Total samples: {len(combined_df)}")
    print(f"Fake news (label=1): {(combined_df['label'] == 1).sum()}")
    print(f"Real news (label=0): {(combined_df['label'] == 0).sum()}")
    
    # Save combined dataset
    output_path = os.path.join(OUT_DIR, 'updated_news_dataset.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"\nCombined dataset saved to: {output_path}")
    
    return combined_df


# ============================================================================
# Preprocessing Module
# ============================================================================


def find_text_col(df: pd.DataFrame) -> str:
    """
    Automatically detect the text column in a DataFrame.
    
    Checks for standard column names in order: 'text', 'title', 'headline', 'content'.
    If none found, falls back to the first string/object column.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Name of the detected text column
    
    Raises:
        ValueError: If no suitable text column is found
    """
    # Check for standard column names in priority order
    standard_names = ['text', 'title', 'headline', 'content']
    for col_name in standard_names:
        if col_name in df.columns:
            print(f"Text column detected: '{col_name}'")
            return col_name
    
    # Fallback: find first string/object column
    for col_name in df.columns:
        if df[col_name].dtype == 'object' or df[col_name].dtype == 'string':
            print(f"Text column detected (fallback): '{col_name}'")
            return col_name
    
    # No suitable column found
    raise ValueError(
        "No suitable text column found in DataFrame. "
        "Expected columns: 'text', 'title', 'headline', 'content', or any string/object column."
    )


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the combined dataset by detecting text column, cleaning, and shuffling.
    
    Args:
        df: Raw combined DataFrame
    
    Returns:
        Cleaned and shuffled DataFrame with standardized 'text' column
    """
    print("\n" + "="*60)
    print("PREPROCESSING DATA")
    print("="*60)
    
    # Detect and rename text column
    text_col = find_text_col(df)
    if text_col != 'text':
        df = df.rename(columns={text_col: 'text'})
        print(f"Renamed column '{text_col}' to 'text'")
    
    # Check initial shape
    print(f"\nInitial dataset shape: {df.shape}")
    print(f"Initial label distribution:\n{df['label'].value_counts().sort_index()}")
    
    # Remove rows with missing values in 'text' or 'label' columns
    initial_count = len(df)
    df = df.dropna(subset=['text', 'label'])
    removed_count = initial_count - len(df)
    
    if removed_count > 0:
        print(f"\nRemoved {removed_count} rows with missing values")
    else:
        print("\nNo missing values found")
    
    # Shuffle dataset with fixed random state for reproducibility
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print("Dataset shuffled with random_state=42")
    
    # Print final statistics
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Final label distribution:\n{df['label'].value_counts().sort_index()}")
    
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.20):
    """
    Split data into train and test sets with stratification.
    
    Args:
        df: Preprocessed DataFrame with 'text' and 'label' columns
        test_size: Proportion for test set (default: 0.20 for 80/20 split)
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print("\n" + "="*60)
    print("SPLITTING DATA")
    print("="*60)
    
    X = df['text']
    y = df['label']
    
    # Stratified split to maintain label distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"\nTrain set size: {len(X_train)} ({(1-test_size)*100:.0f}%)")
    print(f"Test set size: {len(X_test)} ({test_size*100:.0f}%)")
    
    print(f"\nTrain set label distribution:")
    print(y_train.value_counts().sort_index())
    
    print(f"\nTest set label distribution:")
    print(y_test.value_counts().sort_index())
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# Feature Extraction Module
# ============================================================================

def create_tfidf_vectorizer(tfidf_params: dict) -> TfidfVectorizer:
    """
    Create a TF-IDF vectorizer with specified parameters.
    
    Args:
        tfidf_params: Dictionary containing TF-IDF parameters
                     (stop_words, max_df, ngram_range, max_features)
    
    Returns:
        Initialized TfidfVectorizer instance
    """
    vectorizer = TfidfVectorizer(**tfidf_params)
    print(f"\nTF-IDF Vectorizer initialized with parameters:")
    for key, value in tfidf_params.items():
        print(f"  {key}: {value}")
    
    return vectorizer


def fit_transform_vectorizer(vectorizer: TfidfVectorizer, X_train, X_test):
    """
    Fit vectorizer on training set and transform both train and test sets.
    
    Args:
        vectorizer: TfidfVectorizer instance
        X_train: Training text data
        X_test: Test text data
    
    Returns:
        Tuple of (X_train_tfidf, X_test_tfidf) sparse matrices
    """
    print("\n" + "="*60)
    print("FEATURE EXTRACTION")
    print("="*60)
    
    # Fit vectorizer only on training set to prevent data leakage
    print("\nFitting TF-IDF vectorizer on training set...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    print(f"Training set TF-IDF shape: {X_train_tfidf.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Transform test set using fitted vectorizer
    print("\nTransforming test set...")
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Test set TF-IDF shape: {X_test_tfidf.shape}")
    
    return X_train_tfidf, X_test_tfidf


def save_vectorizer(vectorizer: TfidfVectorizer, filepath: str):
    """
    Save fitted vectorizer to disk using joblib.
    
    Args:
        vectorizer: Fitted TfidfVectorizer instance
        filepath: Path where vectorizer should be saved
    """
    joblib.dump(vectorizer, filepath)
    print(f"\nVectorizer saved to: {filepath}")


def load_vectorizer(filepath: str) -> TfidfVectorizer:
    """
    Load fitted vectorizer from disk.
    
    Args:
        filepath: Path to saved vectorizer file
    
    Returns:
        Loaded TfidfVectorizer instance
    """
    vectorizer = joblib.load(filepath)
    print(f"\nVectorizer loaded from: {filepath}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    return vectorizer


def get_or_create_vectorizer(X_train, X_test, tfidf_params: dict, 
                              vectorizer_path: str):
    """
    Load existing vectorizer if available, otherwise create and fit new one.
    
    Args:
        X_train: Training text data
        X_test: Test text data
        tfidf_params: Dictionary containing TF-IDF parameters
        vectorizer_path: Path to vectorizer file
    
    Returns:
        Tuple of (vectorizer, X_train_tfidf, X_test_tfidf)
    """
    if os.path.exists(vectorizer_path):
        print(f"\nFound existing vectorizer at: {vectorizer_path}")
        vectorizer = load_vectorizer(vectorizer_path)
        
        # Transform both train and test sets
        print("\nTransforming training set...")
        X_train_tfidf = vectorizer.transform(X_train)
        print(f"Training set TF-IDF shape: {X_train_tfidf.shape}")
        
        print("\nTransforming test set...")
        X_test_tfidf = vectorizer.transform(X_test)
        print(f"Test set TF-IDF shape: {X_test_tfidf.shape}")
        
    else:
        print(f"\nNo existing vectorizer found. Creating new one...")
        vectorizer = create_tfidf_vectorizer(tfidf_params)
        X_train_tfidf, X_test_tfidf = fit_transform_vectorizer(
            vectorizer, X_train, X_test
        )
        save_vectorizer(vectorizer, vectorizer_path)
    
    return vectorizer, X_train_tfidf, X_test_tfidf


# ============================================================================
# Model Training Module
# ============================================================================

def train_naive_bayes(X_train_tfidf, y_train, model_path: str) -> MultinomialNB:
    """
    Train Multinomial Naive Bayes classifier.
    
    Args:
        X_train_tfidf: TF-IDF transformed training features (sparse matrix)
        y_train: Training labels
        model_path: Path where trained model should be saved
    
    Returns:
        Trained MultinomialNB model
    """
    print("\n" + "="*60)
    print("TRAINING MULTINOMIAL NAIVE BAYES")
    print("="*60)
    
    # Create MultinomialNB instance with default parameters
    model = MultinomialNB()
    
    print("\nTraining Multinomial Naive Bayes model...")
    print(f"Training samples: {X_train_tfidf.shape[0]}")
    print(f"Features: {X_train_tfidf.shape[1]}")
    
    # Fit model on TF-IDF transformed training data
    model.fit(X_train_tfidf, y_train)
    
    print("Training completed!")
    
    # Save trained model
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    return model


def train_logistic_regression(X_train_tfidf, y_train, param_grid: dict, 
                               model_path: str) -> LogisticRegression:
    """
    Train Logistic Regression with GridSearchCV for hyperparameter tuning.
    
    Args:
        X_train_tfidf: TF-IDF transformed training features (sparse matrix)
        y_train: Training labels
        param_grid: Dictionary with hyperparameter grid for search
                   (C, penalty, solver)
        model_path: Path where trained model should be saved
    
    Returns:
        Best LogisticRegression model from grid search
    """
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION WITH GRIDSEARCHCV")
    print("="*60)
    
    # Create LogisticRegression with max_iter=1000 and random_state=42
    base_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    
    print("\nParameter grid for GridSearchCV:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    
    print(f"\nTraining samples: {X_train_tfidf.shape[0]}")
    print(f"Features: {X_train_tfidf.shape[1]}")
    
    # Run GridSearchCV with 4-fold cross-validation, accuracy scoring, and n_jobs=-1
    print("\nRunning GridSearchCV with 4-fold cross-validation...")
    print("This may take a few minutes...")
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=4,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_tfidf, y_train)
    
    # Extract best estimator from grid search results
    best_model = grid_search.best_estimator_
    
    # Print best hyperparameters to console
    print("\n" + "="*60)
    print("GRIDSEARCHCV RESULTS")
    print("="*60)
    print(f"\nBest hyperparameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest cross-validation accuracy: {grid_search.best_score_:.4f}")
    print("Training completed!")
    
    # Save optimized model
    joblib.dump(best_model, model_path)
    print(f"Model saved to: {model_path}")
    
    return best_model


def load_model(model_path: str):
    """
    Load trained model from disk using joblib.
    
    Args:
        model_path: Path to saved model file
    
    Returns:
        Loaded model instance
    """
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    return model


def get_or_train_naive_bayes(X_train_tfidf, y_train, model_path: str) -> MultinomialNB:
    """
    Load existing Naive Bayes model if available, otherwise train new one.
    
    Args:
        X_train_tfidf: TF-IDF transformed training features
        y_train: Training labels
        model_path: Path to model file
    
    Returns:
        MultinomialNB model (loaded or newly trained)
    """
    if os.path.exists(model_path):
        print(f"\nFound existing Naive Bayes model at: {model_path}")
        model = load_model(model_path)
        print("Skipping training step (model loaded from disk)")
    else:
        print(f"\nNo existing Naive Bayes model found. Training new model...")
        model = train_naive_bayes(X_train_tfidf, y_train, model_path)
    
    return model


def get_or_train_logistic_regression(X_train_tfidf, y_train, param_grid: dict,
                                      model_path: str) -> LogisticRegression:
    """
    Load existing Logistic Regression model if available, otherwise train new one.
    
    Args:
        X_train_tfidf: TF-IDF transformed training features
        y_train: Training labels
        param_grid: Dictionary with hyperparameter grid for search
        model_path: Path to model file
    
    Returns:
        LogisticRegression model (loaded or newly trained)
    """
    if os.path.exists(model_path):
        print(f"\nFound existing Logistic Regression model at: {model_path}")
        model = load_model(model_path)
        print("Skipping training step (model loaded from disk)")
    else:
        print(f"\nNo existing Logistic Regression model found. Training new model...")
        model = train_logistic_regression(X_train_tfidf, y_train, param_grid, model_path)
    
    return model


# ============================================================================
# Evaluation Module
# ============================================================================

def get_predictions(model, X_test_tfidf):
    """
    Generate predictions on test set using model.predict().
    
    Args:
        model: Trained classifier (MNB or LR)
        X_test_tfidf: TF-IDF transformed test features (sparse matrix)
    
    Returns:
        Array of predicted labels (0 or 1)
    """
    predictions = model.predict(X_test_tfidf)
    return predictions


def get_probabilities(model, X_test_tfidf):
    """
    Extract probability scores for ROC and Precision-Recall curves.
    
    Uses predict_proba() if available, otherwise falls back to decision_function()
    with min-max scaling to create scores between 0 and 1.
    
    Args:
        model: Trained classifier (MNB or LR)
        X_test_tfidf: TF-IDF transformed test features (sparse matrix)
    
    Returns:
        Array of probability scores for positive class (fake news = 1)
    """
    # Try predict_proba first (available for most sklearn classifiers)
    if hasattr(model, 'predict_proba'):
        # Get probabilities for positive class (index 1)
        probabilities = model.predict_proba(X_test_tfidf)[:, 1]
        return probabilities
    
    # Fallback to decision_function with min-max scaling
    elif hasattr(model, 'decision_function'):
        # Get decision function scores
        decision_scores = model.decision_function(X_test_tfidf)
        
        # Apply min-max scaling to convert to [0, 1] range
        min_score = decision_scores.min()
        max_score = decision_scores.max()
        
        if max_score - min_score > 0:
            probabilities = (decision_scores - min_score) / (max_score - min_score)
        else:
            # If all scores are the same, return 0.5 for all samples
            probabilities = np.full(len(decision_scores), 0.5)
        
        return probabilities
    
    else:
        raise ValueError("Model does not have predict_proba or decision_function method")


def calculate_metrics(y_true, y_pred) -> dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Calculates accuracy, precision, recall, and F1-score using binary, macro,
    and weighted averaging methods.
    
    Args:
        y_true: True labels from test set
        y_pred: Predicted labels from model
    
    Returns:
        Dictionary containing all metrics in structured format:
        {
            'accuracy': float,
            'binary': {'precision': float, 'recall': float, 'f1': float},
            'macro': {'precision': float, 'recall': float, 'f1': float},
            'weighted': {'precision': float, 'recall': float, 'f1': float}
        }
    """
    # Calculate accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # Calculate binary metrics (for positive class = 1, fake news)
    binary_precision = precision_score(y_true, y_pred, average='binary')
    binary_recall = recall_score(y_true, y_pred, average='binary')
    binary_f1 = f1_score(y_true, y_pred, average='binary')
    
    # Calculate macro-averaged metrics (treats both classes equally)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # Calculate weighted-averaged metrics (accounts for class imbalance)
    weighted_precision = precision_score(y_true, y_pred, average='weighted')
    weighted_recall = recall_score(y_true, y_pred, average='weighted')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Store all metrics in structured dictionary
    metrics = {
        'accuracy': acc,
        'binary': {
            'precision': binary_precision,
            'recall': binary_recall,
            'f1': binary_f1
        },
        'macro': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        'weighted': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1
        }
    }
    
    return metrics


def print_metrics(model_name: str, metrics: dict):
    """
    Print evaluation metrics to console with 4 decimal places precision.
    
    Args:
        model_name: Name of the model (e.g., "Multinomial Naive Bayes")
        metrics: Dictionary containing all metrics from calculate_metrics()
    """
    print("\n" + "="*60)
    print(f"{model_name.upper()} - EVALUATION METRICS")
    print("="*60)
    
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    
    print("\nBinary Metrics (Positive Class = Fake News):")
    print(f"  Precision: {metrics['binary']['precision']:.4f}")
    print(f"  Recall:    {metrics['binary']['recall']:.4f}")
    print(f"  F1-Score:  {metrics['binary']['f1']:.4f}")
    
    print("\nMacro-Averaged Metrics:")
    print(f"  Precision: {metrics['macro']['precision']:.4f}")
    print(f"  Recall:    {metrics['macro']['recall']:.4f}")
    print(f"  F1-Score:  {metrics['macro']['f1']:.4f}")
    
    print("\nWeighted-Averaged Metrics:")
    print(f"  Precision: {metrics['weighted']['precision']:.4f}")
    print(f"  Recall:    {metrics['weighted']['recall']:.4f}")
    print(f"  F1-Score:  {metrics['weighted']['f1']:.4f}")


def generate_classification_report(model_name: str, y_true, y_pred):
    """
    Generate detailed classification report and save to file.
    
    Uses sklearn.metrics.classification_report to create a comprehensive report
    with per-class metrics and saves it to outputs/{ModelName}_classification_report.txt
    
    Args:
        model_name: Name of the model (e.g., "MultinomialNB", "LogisticRegression")
        y_true: True labels from test set
        y_pred: Predicted labels from model
    """
    # Generate detailed classification report
    report = classification_report(
        y_true, 
        y_pred,
        target_names=['Real News (0)', 'Fake News (1)'],
        digits=4
    )
    
    # Create filename from model name
    filename = f"{model_name}_classification_report.txt"
    filepath = os.path.join(OUT_DIR, filename)
    
    # Save report to file
    with open(filepath, 'w') as f:
        f.write(f"Classification Report for {model_name}\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    
    print(f"\nClassification report saved to: {filepath}")


def calculate_confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix and extract TN, FP, FN, TP values.
    
    Args:
        y_true: True labels from test set
        y_pred: Predicted labels from model
    
    Returns:
        Tuple of (cm, tn, fp, fn, tp) where:
        - cm: confusion matrix as 2D array
        - tn: true negatives (correctly predicted real news)
        - fp: false positives (real news predicted as fake)
        - fn: false negatives (fake news predicted as real)
        - tp: true positives (correctly predicted fake news)
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Extract TN, FP, FN, TP values
    # For binary classification with labels [0, 1]:
    # cm[0, 0] = TN (predicted 0, actual 0)
    # cm[0, 1] = FP (predicted 1, actual 0)
    # cm[1, 0] = FN (predicted 0, actual 1)
    # cm[1, 1] = TP (predicted 1, actual 1)
    tn, fp, fn, tp = cm.ravel()
    
    return cm, tn, fp, fn, tp


def calculate_roc_auc(y_true, y_probs):
    """
    Compute ROC curve and calculate AUC score.
    
    Args:
        y_true: True labels from test set
        y_probs: Probability scores for positive class (fake news = 1)
    
    Returns:
        Tuple of (fpr, tpr, roc_auc) where:
        - fpr: false positive rates
        - tpr: true positive rates (recall)
        - roc_auc: area under the ROC curve
    """
    # Compute ROC curve using roc_curve() function
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    # Calculate AUC score using auc() function
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc


# ============================================================================
# Visualization Module
# ============================================================================

def show_confusion(y_true, y_pred, model_name: str):
    """
    Display confusion matrix plot using ConfusionMatrixDisplay.
    
    Args:
        y_true: True labels from test set
        y_pred: Predicted labels from model
        model_name: Name of the model for plot title
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create ConfusionMatrixDisplay with labels 0=Real, 1=Fake
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Real (0)', 'Fake (1)']
    )
    
    # Plot with blue colormap for clarity
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    
    # Show plot interactively
    plt.show()


def show_roc(y_true, y_probs, model_name: str):
    """
    Display ROC curve using RocCurveDisplay.
    
    Args:
        y_true: True labels from test set
        y_probs: Probability scores for positive class
        model_name: Name of the model for plot title
    """
    # Calculate ROC curve and AUC
    fpr, tpr, roc_auc = calculate_roc_auc(y_true, y_probs)
    
    # Create RocCurveDisplay
    disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    
    # Plot false positive rate vs true positive rate
    disp.plot()
    
    # Display AUC score in plot title
    plt.title(f'ROC Curve - {model_name} (AUC = {roc_auc:.4f})')
    plt.tight_layout()
    
    # Show plot interactively
    plt.show()


def show_pr(y_true, y_probs, model_name: str):
    """
    Display Precision-Recall curve.
    
    Args:
        y_true: True labels from test set
        y_probs: Probability scores for positive class
        model_name: Name of the model for plot title
    """
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Plot recall vs precision
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Show plot interactively
    plt.show()


def show_accuracy_comparison(accuracies: dict):
    """
    Display bar chart comparing test accuracy of both models.
    
    Args:
        accuracies: Dictionary with model names as keys and accuracy values
                   Example: {'Multinomial Naive Bayes': 0.95, 'Logistic Regression': 0.99}
    """
    # Create bar chart
    plt.figure(figsize=(10, 6))
    
    model_names = list(accuracies.keys())
    accuracy_values = list(accuracies.values())
    
    bars = plt.bar(model_names, accuracy_values, color=['skyblue', 'lightcoral'])
    
    # Display accuracy values on top of bars
    for bar, acc in zip(bars, accuracy_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    
    # Set y-axis limit from 0 to 1
    plt.ylim(0, 1)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Show plot interactively
    plt.show()


def make_pipeline_for_learning(vectorizer: TfidfVectorizer, model):
    """
    Create a pipeline for learning curve generation.
    
    Helper function that combines vectorizer and model into a pipeline
    so learning_curve can work with raw text data.
    
    Args:
        vectorizer: Fitted TfidfVectorizer instance
        model: Trained classifier (MNB or LR)
    
    Returns:
        sklearn Pipeline object
    """
    # Create pipeline with vectorizer and model
    pipeline = make_pipeline(vectorizer, model)
    return pipeline


def show_learning_curve(model, vectorizer: TfidfVectorizer, X_train, y_train, 
                        model_name: str):
    """
    Display learning curve showing training and validation accuracy vs training set size.
    
    Args:
        model: Trained classifier (MNB or LR)
        vectorizer: Fitted TfidfVectorizer instance
        X_train: Training text data (raw text, not TF-IDF)
        y_train: Training labels
        model_name: Name of the model for plot title
    """
    print(f"\nGenerating learning curve for {model_name}...")
    
    # Create pipeline for learning curve
    pipeline = make_pipeline_for_learning(vectorizer, model)
    
    # Use learning_curve with 4-fold CV and 5 training size points (10% to 100%)
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline,
        X_train,
        y_train,
        cv=4,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='accuracy',
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    
    # Calculate mean and std for training and validation scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot training accuracy and validation accuracy vs training set size
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Accuracy')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title(f'Learning Curve - {model_name}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Show plot interactively
    plt.show()


def show_error_curve(model, vectorizer: TfidfVectorizer, X_train, y_train, 
                     model_name: str):
    """
    Display error vs training size plot showing training and validation error.
    
    Args:
        model: Trained classifier (MNB or LR)
        vectorizer: Fitted TfidfVectorizer instance
        X_train: Training text data (raw text, not TF-IDF)
        y_train: Training labels
        model_name: Name of the model for plot title
    """
    print(f"\nGenerating error curve for {model_name}...")
    
    # Create pipeline for learning curve
    pipeline = make_pipeline_for_learning(vectorizer, model)
    
    # Use learning_curve with 4-fold CV and 5 training size points (10% to 100%)
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline,
        X_train,
        y_train,
        cv=4,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='accuracy',
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    
    # Calculate training error and validation error (1 - accuracy)
    train_error_mean = 1 - np.mean(train_scores, axis=1)
    train_error_std = np.std(train_scores, axis=1)
    val_error_mean = 1 - np.mean(val_scores, axis=1)
    val_error_std = np.std(val_scores, axis=1)
    
    # Plot both errors vs training set size
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_sizes, train_error_mean, 'o-', color='blue', label='Training Error')
    plt.fill_between(train_sizes, train_error_mean - train_error_std, 
                     train_error_mean + train_error_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_error_mean, 'o-', color='red', label='Validation Error')
    plt.fill_between(train_sizes, val_error_mean - val_error_std, 
                     val_error_mean + val_error_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Error (1 - Accuracy)')
    plt.title(f'Error vs Training Size - {model_name}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Show plot interactively
    plt.show()


def display_all_plots(models_dict: dict, vectorizer: TfidfVectorizer, 
                      X_train, y_train, X_test_tfidf, y_test):
    """
    Display all plots for both models.
    
    Loops through both MNB and LR models and generates all visualizations:
    - Confusion matrix
    - ROC curve
    - Precision-Recall curve
    - Learning curves
    - Error curves
    
    Handles plotting exceptions gracefully with try-except.
    
    Args:
        models_dict: Dictionary with model names as keys and model objects as values
                    Example: {'Multinomial Naive Bayes': mnb_model, 
                             'Logistic Regression': lr_model}
        vectorizer: Fitted TfidfVectorizer instance
        X_train: Training text data (raw text, not TF-IDF)
        y_train: Training labels
        X_test_tfidf: Test features (TF-IDF transformed)
        y_test: Test labels
    """
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Loop through both MNB and LR models
    for model_name, model in models_dict.items():
        print(f"\n{'='*60}")
        print(f"Visualizations for {model_name}")
        print(f"{'='*60}")
        
        try:
            # Get predictions and probabilities
            y_pred = get_predictions(model, X_test_tfidf)
            y_probs = get_probabilities(model, X_test_tfidf)
            
            # Generate confusion matrix
            print(f"\nDisplaying confusion matrix for {model_name}...")
            show_confusion(y_test, y_pred, model_name)
            
            # Generate ROC curve
            print(f"Displaying ROC curve for {model_name}...")
            show_roc(y_test, y_probs, model_name)
            
            # Generate Precision-Recall curve
            print(f"Displaying Precision-Recall curve for {model_name}...")
            show_pr(y_test, y_probs, model_name)
            
            # Generate learning curve
            print(f"Displaying learning curve for {model_name}...")
            show_learning_curve(model, vectorizer, X_train, y_train, model_name)
            
            # Generate error curve
            print(f"Displaying error curve for {model_name}...")
            show_error_curve(model, vectorizer, X_train, y_train, model_name)
            
            print(f"\nAll visualizations completed for {model_name}")
            
        except Exception as e:
            print(f"\nError generating visualizations for {model_name}: {e}")
            print("Continuing with next model...")
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS COMPLETED")
    print("="*60)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("FAKE NEWS DETECTION SYSTEM")
    print("Created by Mehul Gupta")
    print("="*60)
    print(f"Output directory: {OUT_DIR}")
    print(f"Random state: {RANDOM_STATE}")
    print("="*60)
    
    # ========================================================================
    # TASK 8.1: Wire data collection to preprocessing
    # ========================================================================
    
    # Load datasets from CSV files
    fake_df, true_df = load_datasets(FAKE_CSV, TRUE_CSV)
    
    # Call web scraping functions to augment data
    scraped_df = get_all_real_news()
    
    # Combine all data sources and save to CSV
    combined_df = aggregate_all_data(fake_df, true_df, scraped_df)
    
    # Pass combined DataFrame to preprocessing functions
    preprocessed_df = preprocess_data(combined_df)
    
    # ========================================================================
    # TASK 8.2: Wire preprocessing to feature extraction
    # ========================================================================
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = split_data(preprocessed_df)
    
    # Pass text data to TF-IDF vectorizer
    vectorizer_path = os.path.join(OUT_DIR, 'tfidf_vectorizer.joblib')
    vectorizer, X_train_tfidf, X_test_tfidf = get_or_create_vectorizer(
        X_train, X_test, TFIDF_PARAMS, vectorizer_path
    )
    
    # ========================================================================
    # TASK 8.3: Wire feature extraction to model training
    # ========================================================================
    # (TF-IDF matrices already created in task 8.2)
    
    # ========================================================================
    # TASK 8.4: Wire model training to evaluation
    # ========================================================================
    
    # Train or load both MNB and LR models
    mnb_path = os.path.join(OUT_DIR, 'naive_bayes_model.joblib')
    mnb_model = get_or_train_naive_bayes(X_train_tfidf, y_train, mnb_path)
    
    lr_path = os.path.join(OUT_DIR, 'logistic_regression_model_grid.joblib')
    lr_model = get_or_train_logistic_regression(X_train_tfidf, y_train, GRID_PARAMS, lr_path)
    
    # Pass trained models and test data to evaluation functions
    # Collect metrics for both models
    models = {
        'Multinomial Naive Bayes': mnb_model,
        'Logistic Regression': lr_model
    }
    
    all_metrics = {}
    
    for model_name, model in models.items():
        # Get predictions
        y_pred = get_predictions(model, X_test_tfidf)
        
        # Get probabilities
        y_probs = get_probabilities(model, X_test_tfidf)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred)
        all_metrics[model_name] = metrics
        
        # Print metrics
        print_metrics(model_name, metrics)
        
        # Generate classification report
        model_filename = model_name.replace(' ', '')
        generate_classification_report(model_filename, y_test, y_pred)
        
        # Calculate confusion matrix
        cm, tn, fp, fn, tp = calculate_confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix Values:")
        print(f"  True Negatives (TN):  {tn}")
        print(f"  False Positives (FP): {fp}")
        print(f"  False Negatives (FN): {fn}")
        print(f"  True Positives (TP):  {tp}")
        
        # Calculate ROC AUC
        fpr, tpr, roc_auc = calculate_roc_auc(y_test, y_probs)
        print(f"\nROC AUC Score: {roc_auc:.4f}")
    
    # ========================================================================
    # TASK 8.5: Wire evaluation to visualization
    # ========================================================================
    
    # Display accuracy comparison
    accuracies = {name: metrics['accuracy'] for name, metrics in all_metrics.items()}
    show_accuracy_comparison(accuracies)
    
    # Display all plots for both models
    display_all_plots(models, vectorizer, X_train, y_train, X_test_tfidf, y_test)
    
    # Display final metrics summary
    print("\n" + "="*60)
    print("FINAL METRICS SUMMARY")
    print("="*60)
    
    for model_name, metrics in all_metrics.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Binary F1-Score: {metrics['binary']['f1']:.4f}")
        print(f"  Macro F1-Score: {metrics['macro']['f1']:.4f}")
    
    print("\n" + "="*60)
    print("FAKE NEWS DETECTION SYSTEM COMPLETED")
    print("="*60)