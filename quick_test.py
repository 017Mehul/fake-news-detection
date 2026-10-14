"""Small smoke test for trained models; run training first."""
from fake_news_detection import predict_news

SAMPLES = [
    "The government announced a new public transport policy today, according to the official press release.",
    "BREAKING: A secret miracle cure has been hidden by every doctor and will fix all diseases overnight!!!",
]


def main():
    for i, text in enumerate(SAMPLES, start=1):
        result = predict_news(text, "logistic_regression")
        print(f"Sample {i}: {result['label']} (confidence={result.get('confidence', 0):.3f})")


if __name__ == "__main__":
    main()
