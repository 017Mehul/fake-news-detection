# Browser-only model setup (no prediction backend)

This project can run inference in the browser, similar to the ASL project's TensorFlow.js model folder. The browser does not execute the Python `.joblib` file directly, so the trained TF-IDF + Logistic Regression pipeline must be exported into a JSON representation first.

## Export the trained model

Keep `Fake.csv`, `True.csv`, and the trained `outputs/*.joblib` artifacts local; they are intentionally git-ignored.

From the repository root, run:

```bash
pip install -r requirements.txt
python fake_news_detection.py train
python export_browser_model.py
```

The export script creates `web/model/model.json`. This contains the fitted vocabulary, IDF values, logistic-regression coefficients, and intercept required for inference. It does not contain the source dataset.

## Add the model file to GitHub

Review the generated `web/model/model.json`, then commit and push that one generated model file. The browser downloads it from the same static site. Do not commit `Fake.csv`, `True.csv`, or the original `.joblib` artifacts.

## Deploy

Deploy the `web/` directory as a static site (for example, Vercel Root Directory = `web`, Framework Preset = Other, build/install commands blank, output directory `.`). The UI's `web/config.js` points to `./model/model.json`; no API URL is required.

Open the deployed site over HTTPS and test an article. If the model file is missing, the UI reports that explicitly rather than showing a fake demo result.

## Notes

- The first prediction downloads the model JSON; subsequent predictions reuse it in that page session.
- Model confidence is the classifier's estimated probability, not a measure of verified truth.
- Browser-side inference keeps submitted article text on the user's device.
- Because the vocabulary and coefficients are shipped to the browser, users can inspect or download them. Do not use this approach for proprietary model weights.
