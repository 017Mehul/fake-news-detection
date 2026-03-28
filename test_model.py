"""
Test the trained fake news detection model on sample articles
"""

import joblib
import pandas as pd

# Load the trained models
print("Loading trained models...")
vectorizer = joblib.load('outputs/tfidf_vectorizer.joblib')
nb_model = joblib.load('outputs/naive_bayes_model.joblib')
lr_model = joblib.load('outputs/logistic_regression_model_grid.joblib')

print("Models loaded successfully!\n")

# Test articles
test_articles = [
    {
        "text": "Scientists at NASA have confirmed that the Earth is flat and all previous evidence was fabricated. The space agency admitted they have been lying for decades to justify their budget.",
        "expected": "FAKE"
    },
    {
        "text": "The Federal Reserve announced today that it will raise interest rates by 0.25 percentage points to combat inflation. The decision was made after careful analysis of economic indicators.",
        "expected": "REAL"
    },
    {
        "text": "Breaking: Hillary Clinton arrested in secret military operation. Deep state operatives are being rounded up across the country. Mainstream media refuses to report this.",
        "expected": "FAKE"
    },
    {
        "text": "Apple Inc. reported quarterly earnings that exceeded analyst expectations. The company's revenue grew 12% year-over-year, driven by strong iPhone sales in international markets.",
        "expected": "REAL"
    },
    {
        "text": "Drinking bleach cures cancer and COVID-19, according to leaked documents from pharmaceutical companies. Big Pharma doesn't want you to know this simple cure.",
        "expected": "FAKE"
    },
    {
        "text": "The Supreme Court ruled today in a 6-3 decision on a case involving voting rights. Chief Justice Roberts wrote the majority opinion, which was joined by five other justices.",
        "expected": "REAL"
    },
    {
        "text": "5G cell towers are spreading coronavirus by weakening immune systems. Government officials have finally admitted the connection after months of cover-up.",
        "expected": "FAKE"
    },
    {
        "text": "The unemployment rate fell to 3.7% last month, according to data released by the Bureau of Labor Statistics. Economists say the labor market remains strong despite recent concerns.",
        "expected": "REAL"
    }
]

print("="*80)
print("TESTING FAKE NEWS DETECTION MODEL")
print("Created by Mehul Gupta")
print("="*80)

# Test with both models
for i, article in enumerate(test_articles, 1):
    print(f"\n{'='*80}")
    print(f"TEST ARTICLE #{i}")
    print(f"{'='*80}")
    print(f"Text: {article['text'][:150]}...")
    print(f"\nExpected: {article['expected']}")
    
    # Transform text to TF-IDF
    text_tfidf = vectorizer.transform([article['text']])
    
    # Predict with Naive Bayes
    nb_pred = nb_model.predict(text_tfidf)[0]
    nb_prob = nb_model.predict_proba(text_tfidf)[0]
    nb_result = "FAKE" if nb_pred == 1 else "REAL"
    
    # Predict with Logistic Regression
    lr_pred = lr_model.predict(text_tfidf)[0]
    lr_prob = lr_model.predict_proba(text_tfidf)[0]
    lr_result = "FAKE" if lr_pred == 1 else "REAL"
    
    print(f"\nNaive Bayes Prediction: {nb_result}")
    print(f"  Confidence - Real: {nb_prob[0]:.2%}, Fake: {nb_prob[1]:.2%}")
    print(f"  Result: {'✓ CORRECT' if nb_result == article['expected'] else '✗ WRONG'}")
    
    print(f"\nLogistic Regression Prediction: {lr_result}")
    print(f"  Confidence - Real: {lr_prob[0]:.2%}, Fake: {lr_prob[1]:.2%}")
    print(f"  Result: {'✓ CORRECT' if lr_result == article['expected'] else '✗ WRONG'}")

print(f"\n{'='*80}")
print("TESTING COMPLETED")
print(f"{'='*80}")

# Calculate accuracy
nb_correct = sum(1 for article in test_articles 
                 if ("FAKE" if nb_model.predict(vectorizer.transform([article['text']]))[0] == 1 else "REAL") == article['expected'])
lr_correct = sum(1 for article in test_articles 
                 if ("FAKE" if lr_model.predict(vectorizer.transform([article['text']]))[0] == 1 else "REAL") == article['expected'])

print(f"\nNaive Bayes: {nb_correct}/{len(test_articles)} correct ({nb_correct/len(test_articles)*100:.1f}%)")
print(f"Logistic Regression: {lr_correct}/{len(test_articles)} correct ({lr_correct/len(test_articles)*100:.1f}%)")
