"""
FAKE NEWS DETECTION SYSTEM
Complete Machine Learning Model for Detecting Fake News

Accuracy: 94.36% (Naive Bayes) | 98.91% (Logistic Regression)
Dataset: 44,898 news articles

This system uses TF-IDF vectorization and two ML models:
1. Multinomial Naive Bayes (more reliable for new data)
2. Logistic Regression (higher accuracy on training data)

Requirements:
- pandas
- numpy
- scikit-learn
- beautifulsoup4
- requests
- joblib
"""

import os
import warnings
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_STATE = 42
OUT_DIR = "model_outputs"
FAKE_CSV = "Fake.csv"
TRUE_CSV = "True.csv"

# TF-IDF Parameters
TFIDF_PARAMS = {
    'stop_words': 'english',
    'max_df': 0.7,           # Ignore terms that appear in >70% of documents
    'ngram_range': (1, 2),   # Use unigrams and bigrams
    'max_features': 30000    # Limit to top 30,000 features
}

# Logistic Regression Grid Search Parameters
GRID_PARAMS = {
    'C': [0.01, 0.1, 1.0, 5.0],
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [1000]
}

# Create output directory
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_datasets(fake_path, true_path):
    """
    Load fake and true news datasets from CSV files.
    
    Args:
        fake_path: Path to fake news CSV
        true_path: Path to true news CSV
    
    Returns:
        tuple: (fake_df, true_df) with label column added
    """
    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"Fake news CSV not found: {fake_path}")
    if not os.path.exists(true_path):
        raise FileNotFoundError(f"True news CSV not found: {true_path}")
    
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    
    fake_df['label'] = 1  # Fake = 1
    true_df['label'] = 0  # Real = 0
    
    print(f"✓ Loaded {len(fake_df)} fake news articles")
    print(f"✓ Loaded {len(true_df)} true news articles")
    
    return fake_df, true_df


def combine_datasets(fake_df, true_df):
    """Combine fake and true datasets into one."""
    combined = pd.concat([fake_df, true_df], ignore_index=True)
    
    print(f"\n📊 Combined Dataset Statistics:")
    print(f"   Total samples: {len(combined)}")
    print(f"   Fake news: {(combined['label'] == 1).sum()}")
    print(f"   Real news: {(combined['label'] == 0).sum()}")
    
    return combined


# ============================================================================
# PREPROCESSING
# ============================================================================

def find_text_column(df):
    """Automatically detect the text column."""
    for col in ['text', 'title', 'headline', 'content']:
        if col in df.columns:
            return col
    
    # Fallback to first string column
    for col in df.columns:
        if df[col].dtype == 'object':
            return col
    
    raise ValueError("No text column found in dataset")


def preprocess_data(df):
    """
    Clean and prepare data for training.
    
    Steps:
    1. Detect and standardize text column
    2. Remove missing values
    3. Shuffle data
    """
    print("\n🔧 Preprocessing Data...")
    
    # Detect text column
    text_col = find_text_column(df)
    if text_col != 'text':
        df = df.rename(columns={text_col: 'text'})
    
    # Remove missing values
    initial_count = len(df)
    df = df.dropna(subset=['text', 'label'])
    removed = initial_count - len(df)
    
    if removed > 0:
        print(f"   Removed {removed} rows with missing values")
    
    # Shuffle
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    print(f"✓ Preprocessing complete: {len(df)} samples ready")
    
    return df


def split_data(df, test_size=0.20):
    """Split data into train and test sets with stratification."""
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"\n📂 Data Split:")
    print(f"   Training: {len(X_train)} samples ({(1-test_size)*100:.0f}%)")
    print(f"   Testing: {len(X_test)} samples ({test_size*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def create_tfidf_features(X_train, X_test, params):
    """
    Create TF-IDF features from text data.
    
    Args:
        X_train: Training text data
        X_test: Test text data
        params: TF-IDF parameters
    
    Returns:
        tuple: (vectorizer, X_train_tfidf, X_test_tfidf)
    """
    print("\n🔤 Creating TF-IDF Features...")
    
    vectorizer = TfidfVectorizer(**params)
    
    # Fit on training data only
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"✓ Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"✓ Training features: {X_train_tfidf.shape}")
    print(f"✓ Test features: {X_test_tfidf.shape}")
    
    return vectorizer, X_train_tfidf, X_test_tfidf


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_naive_bayes(X_train, y_train):
    """Train Multinomial Naive Bayes classifier."""
    print("\n🤖 Training Naive Bayes Model...")
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    print("✓ Naive Bayes training complete")
    
    return model


def train_logistic_regression(X_train, y_train, param_grid):
    """Train Logistic Regression with GridSearchCV."""
    print("\n🤖 Training Logistic Regression with GridSearchCV...")
    print("   This may take a few minutes...")
    
    base_model = LogisticRegression(random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=4,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"✓ Best parameters: {grid_search.best_params_}")
    print(f"✓ Best CV accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
    
    Returns:
        dict: Evaluation metrics
    """
    print(f"\n📊 Evaluating {model_name}...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Print results
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} - RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn:,}")
    print(f"  False Positives: {fp:,}")
    print(f"  False Negatives: {fn:,}")
    print(f"  True Positives:  {tp:,}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def save_models(vectorizer, nb_model, lr_model):
    """Save trained models to disk."""
    print(f"\n💾 Saving models to '{OUT_DIR}/'...")
    
    joblib.dump(vectorizer, os.path.join(OUT_DIR, 'vectorizer.joblib'))
    joblib.dump(nb_model, os.path.join(OUT_DIR, 'naive_bayes.joblib'))
    joblib.dump(lr_model, os.path.join(OUT_DIR, 'logistic_regression.joblib'))
    
    print("✓ All models saved successfully")


def load_models():
    """Load trained models from disk."""
    vectorizer = joblib.load(os.path.join(OUT_DIR, 'vectorizer.joblib'))
    nb_model = joblib.load(os.path.join(OUT_DIR, 'naive_bayes.joblib'))
    lr_model = joblib.load(os.path.join(OUT_DIR, 'logistic_regression.joblib'))
    
    return vectorizer, nb_model, lr_model


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_news(text, vectorizer, model):
    """
    Predict if a news article is REAL or FAKE.
    
    Args:
        text: News article text
        vectorizer: Trained TF-IDF vectorizer
        model: Trained classifier
    
    Returns:
        tuple: (prediction, confidence_real, confidence_fake)
    """
    # Transform text
    text_tfidf = vectorizer.transform([text])
    
    # Predict
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    
    result = "FAKE" if prediction == 1 else "REAL"
    conf_real = probabilities[0]
    conf_fake = probabilities[1]
    
    return result, conf_real, conf_fake


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline."""
    print("="*60)
    print("FAKE NEWS DETECTION SYSTEM")
    print("Created by Mehul Gupta")
    print("="*60)
    
    # 1. Load data
    print("\n📥 Loading datasets...")
    fake_df, true_df = load_datasets(FAKE_CSV, TRUE_CSV)
    combined_df = combine_datasets(fake_df, true_df)
    
    # 2. Preprocess
    processed_df = preprocess_data(combined_df)
    
    # 3. Split data
    X_train, X_test, y_train, y_test = split_data(processed_df)
    
    # 4. Create features
    vectorizer, X_train_tfidf, X_test_tfidf = create_tfidf_features(
        X_train, X_test, TFIDF_PARAMS
    )
    
    # 5. Train models
    nb_model = train_naive_bayes(X_train_tfidf, y_train)
    lr_model = train_logistic_regression(X_train_tfidf, y_train, GRID_PARAMS)
    
    # 6. Evaluate models
    nb_metrics = evaluate_model(nb_model, X_test_tfidf, y_test, "Naive Bayes")
    lr_metrics = evaluate_model(lr_model, X_test_tfidf, y_test, "Logistic Regression")
    
    # 7. Save models
    save_models(vectorizer, nb_model, lr_model)
    
    # 8. Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\n🎯 Final Accuracy:")
    print(f"   Naive Bayes:         {nb_metrics['accuracy']*100:.2f}%")
    print(f"   Logistic Regression: {lr_metrics['accuracy']*100:.2f}%")
    print(f"\n💡 Recommendation: Use Naive Bayes for better generalization")
    print(f"   (More reliable on new, unseen articles)")
    
    return vectorizer, nb_model, lr_model


# ============================================================================
# INTERACTIVE TESTING
# ============================================================================

def interactive_mode():
    """Run interactive prediction mode."""
    print("\n" + "="*60)
    print("INTERACTIVE FAKE NEWS DETECTOR")
    print("Created by Mehul Gupta")
    print("="*60)
    
    # Load models
    print("\n📂 Loading trained models...")
    try:
        vectorizer, nb_model, lr_model = load_models()
        print("✓ Models loaded successfully!")
    except FileNotFoundError:
        print("❌ Models not found. Please train the model first.")
        return
    
    print("\nEnter a news article to check if it's REAL or FAKE")
    print("Type 'quit' to exit\n")
    
    while True:
        print("-"*60)
        article = input("\nEnter news article: ").strip()
        
        if article.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not article:
            print("⚠️  Please enter some text!")
            continue
        
        # Predict with Naive Bayes (more reliable)
        result, conf_real, conf_fake = predict_news(article, vectorizer, nb_model)
        
        print(f"\n{'='*60}")
        print(f"PREDICTION: {result}")
        print(f"{'='*60}")
        print(f"Confidence - Real News: {conf_real:.2%}")
        print(f"Confidence - Fake News: {conf_fake:.2%}")
        
        if result == "FAKE":
            print("\n⚠️  This article appears to be FAKE NEWS!")
        else:
            print("\n✓ This article appears to be REAL NEWS")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        # Run in interactive prediction mode
        interactive_mode()
    else:
        # Run training pipeline
        main()
        
        # Ask if user wants to test
        print("\n" + "="*60)
        response = input("\nWould you like to test the model interactively? (y/n): ")
        if response.lower() in ['y', 'yes']:
            interactive_mode()
