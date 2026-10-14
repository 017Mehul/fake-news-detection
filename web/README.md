# Fake News Detection — Browser App

A responsive frontend that classifies pasted news article text as **Likely Real News** or **Likely Fake News** using a TF-IDF + Logistic Regression model exported to JSON and run locally in the browser.

## Run locally

From the repository root, start a static server:

```bash
python -m http.server 8000
```

Open [http://localhost:8000/web/](http://localhost:8000/web/).

You can also deploy the `web/` directory to a static hosting provider. Keep the relative model path intact, or update `window.FAKE_NEWS_MODEL_URL` in `config.js` if your hosting layout requires a different path. Opening the HTML file directly with `file://` may prevent the browser from loading the model JSON; use a local server instead.

## How it works

- `index.html`, `styles.css`, and `script.js` provide the interface and prediction flow.
- `config.js` points to `./model/model.json`.
- `model/model.json` contains the exported TF-IDF vocabulary/IDF values and Logistic Regression weights used for in-browser inference.
- The article text is processed in the browser; this demo does not require a prediction API or server-side model endpoint.

## Prediction steps

1. Paste article text into the input area.
2. Submit the article for analysis.
3. Review the predicted label and confidence shown by the interface.

The app validates input length and displays an error if the model cannot be loaded. If you change the model export, rerun the exporter and verify the browser predictions against the matching Python pipeline.

## Model validation

The exported browser model was checked against its matching Python pipeline on 16 sample articles, with matching predicted labels and probabilities within floating-point tolerance. A separate held-out training evaluation reported 98.42% accuracy for that training run. These are offline results, not a guarantee of real-world performance.

## Important limitations

- This is a text classifier, **not a fact-checking service**. A prediction does not prove a claim true or false.
- Test examples and model confidence are not independent verification of an article.
- Performance can vary on current events, new topics, different publishers, satire, and text unlike the training data.
- Do not use the result as the sole basis for important decisions.

See the repository's root [README](../README.md) for training, CLI prediction, tests, dataset setup, and project limitations.
