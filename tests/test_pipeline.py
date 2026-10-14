"""Unit tests for data validation and prediction input handling."""
import unittest
from unittest.mock import patch
import pandas as pd
from pathlib import Path
from fake_news_detection import load_dataset, predict_news


class DatasetTests(unittest.TestCase):
    def test_missing_dataset_is_actionable(self):
        with self.assertRaisesRegex(FileNotFoundError, "DATASET.md"):
            load_dataset(Path("definitely_missing_fake.csv"), Path("definitely_missing_true.csv"))

    def test_missing_text_column_is_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "Fake.csv"
            real = Path(tmp) / "True.csv"
            pd.DataFrame({"title": ["x"]}).to_csv(fake, index=False)
            pd.DataFrame({"text": ["y"]}).to_csv(real, index=False)
            with self.assertRaisesRegex(ValueError, "text"):
                load_dataset(fake, real)

    def test_empty_prediction_is_rejected_before_loading_model(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            predict_news("   ")


if __name__ == "__main__":
    unittest.main()
