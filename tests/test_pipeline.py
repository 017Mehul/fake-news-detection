"""Unit tests for data validation and prediction input handling."""
import unittest
from unittest.mock import patch
import pandas as pd
from pathlib import Path
import fake_news_detection as detector
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


    def test_training_integration_creates_models_metrics_and_prediction(self):
        import tempfile
        from unittest.mock import patch

        texts = []
        labels = []
        for i in range(12):
            texts.append(f"fabricated fake scandal report invented claim tokenfake{i}")
            labels.append(1)
            texts.append(f"verified real official report confirmed statement tokenreal{i}")
            labels.append(0)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            nb_path = output_dir / "naive_bayes_model.joblib"
            lr_path = output_dir / "logistic_regression_model_grid.joblib"
            metrics_path = output_dir / "metrics.json"
            with patch.object(detector, "load_dataset", return_value=(pd.Series(texts), pd.Series(labels))), \
                 patch.object(detector, "OUT_DIR", output_dir), \
                 patch.object(detector, "NAIVE_BAYES_PATH", nb_path), \
                 patch.object(detector, "LOGISTIC_PATH", lr_path), \
                 patch.object(detector, "METRICS_PATH", metrics_path):
                metrics = detector.train_and_evaluate()
                self.assertTrue(nb_path.is_file())
                self.assertTrue(lr_path.is_file())
                self.assertTrue(metrics_path.is_file())
                self.assertIn("logistic_regression", metrics["models"])
                prediction = detector.predict_news("verified official report confirmed statement", "logistic_regression")
                self.assertIn(prediction["label"], {"REAL", "FAKE"})
                self.assertIn("confidence", prediction)


if __name__ == "__main__":
    unittest.main()
