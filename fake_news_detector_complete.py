"""Backward-compatible entry point.

Use fake_news_detection.py as the canonical implementation. This file remains
for users following older instructions and no longer maintains separate model
paths or artifacts.
"""
import argparse
from fake_news_detection import train_and_evaluate, predict_news


def main():
    parser = argparse.ArgumentParser(description="Legacy entry point for fake-news detection.")
    parser.add_argument("--predict", help="Optional article text to classify.")
    parser.add_argument("--model", choices=["naive_bayes", "logistic_regression"], default="logistic_regression")
    args = parser.parse_args()
    if args.predict is not None:
        print(predict_news(args.predict, args.model))
    else:
        train_and_evaluate()


if __name__ == "__main__":
    main()
