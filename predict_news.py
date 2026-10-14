"""Simple command-line prediction helper. Train first with fake_news_detection.py."""
import argparse
import json
from fake_news_detection import predict_news


def main():
    parser = argparse.ArgumentParser(description="Predict whether a news article is fake or real.")
    parser.add_argument("text", help="Full article text in quotes.")
    parser.add_argument("--model", choices=["naive_bayes", "logistic_regression"], default="logistic_regression")
    args = parser.parse_args()
    try:
        print(json.dumps(predict_news(args.text, args.model), indent=2))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
