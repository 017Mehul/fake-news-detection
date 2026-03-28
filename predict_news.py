"""
Simple prediction script using the BEST performing model (Naive Bayes)
"""

import joblib

# Load the models
print("Loading models...")
vectorizer = joblib.load('outputs/tfidf_vectorizer.joblib')
model = joblib.load('outputs/naive_bayes_model.joblib')  # Using Naive Bayes - it's more reliable!
print("✓ Model loaded successfully!\n")

def predict_news(text):
    """
    Predict if a news article is REAL or FAKE
    
    Args:
        text: The news article text
    
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

# Interactive mode
print("="*80)
print("FAKE NEWS DETECTOR (Using Naive Bayes - 94.4% Accuracy)")
print("Created by Mehul Gupta")
print("="*80)
print("\nEnter a news article to check if it's REAL or FAKE")

print("Type 'quit' to exit\n")

while True:
    print("-"*80)
    article = input("\nEnter news article: ").strip()
    
    if article.lower() in ['quit', 'exit', 'q']:
        print("\nGoodbye!")
        break
    
    if not article:
        print("Please enter some text!")
        continue
    
    result, conf_real, conf_fake = predict_news(article)
    
    print(f"\n{'='*80}")
    print(f"PREDICTION: {result}")
    print(f"{'='*80}")
    print(f"Confidence - Real News: {conf_real:.2%}")
    print(f"Confidence - Fake News: {conf_fake:.2%}")
    
    if result == "FAKE":
        print("\n⚠️  This article appears to be FAKE NEWS!")
    else:
        print("\n✓ This article appears to be REAL NEWS")
