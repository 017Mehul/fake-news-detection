"""
Quick Test Script for Fake News Detection Model
Run this after training to verify the model works correctly.
Loads models from the outputs/ directory produced by fake_news_detection.py
"""

import joblib

# Test articles
test_cases = [
    {
        "text": "Scientists at NASA confirm Earth is flat, all previous evidence was fabricated",
        "expected": "FAKE"
    },
    {
        "text": "The Federal Reserve announced today it will raise interest rates by 0.25 percentage points",
        "expected": "REAL"
    },
    {
        "text": "Hillary Clinton arrested in secret military operation, deep state exposed",
        "expected": "FAKE"
    },
    {
        "text": "Apple Inc reported quarterly earnings that exceeded analyst expectations",
        "expected": "REAL"
    }
]


def run_tests():
    """Run quick tests on the trained model."""
    print("="*70)
    print("QUICK TEST - FAKE NEWS DETECTION MODEL")
    print("Created by Mehul Gupta")
    print("="*70)

    # Load models from outputs/ (produced by fake_news_detection.py)
    print("\nLoading models...")
    try:
        vectorizer = joblib.load('outputs/tfidf_vectorizer.joblib')
        nb_model = joblib.load('outputs/naive_bayes_model.joblib')
        print("Models loaded successfully!\n")
    except FileNotFoundError:
        print("Error: Models not found in outputs/")
        print("Please train the model first by running:")
        print("   python fake_news_detection.py")
        return

    # Run tests
    correct = 0
    total = len(test_cases)

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{total}")
        print(f"{'='*70}")
        print(f"Article: {test['text'][:80]}...")
        print(f"Expected: {test['expected']}")

        # Transform and predict
        text_tfidf = vectorizer.transform([test['text']])
        prediction = nb_model.predict(text_tfidf)[0]
        probabilities = nb_model.predict_proba(text_tfidf)[0]
        result = "FAKE" if prediction == 1 else "REAL"

        # Check result
        is_correct = result == test['expected']
        if is_correct:
            correct += 1

        print(f"\nPrediction: {result}")
        print(f"Confidence: Real={probabilities[0]:.1%}, Fake={probabilities[1]:.1%}")
        print(f"Result: {'CORRECT' if is_correct else 'WRONG'}")

    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Passed: {correct}/{total} ({correct/total*100:.0f}%)")

    if correct == total:
        print("\nAll tests passed! Model is working correctly.")
    elif correct >= total * 0.75:
        print("\nMost tests passed. Model is working reasonably well.")
    else:
        print("\nMany tests failed. Model may need retraining.")


if __name__ == "__main__":
    run_tests()
