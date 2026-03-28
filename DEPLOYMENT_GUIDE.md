# Fake News Detection Model - Deployment Guide

This guide will help you deploy and use the fake news detection model anywhere.

## 📦 What You Need to Upload

### Essential Files:
1. **`fake_news_detector_complete.py`** - Main model code (complete system)
2. **`model_requirements.txt`** - Python dependencies
3. **`MODEL_README.md`** - Documentation
4. **`quick_test.py`** - Testing script

### Model Files (after training):
5. **`model_outputs/vectorizer.joblib`** - TF-IDF vectorizer
6. **`model_outputs/naive_bayes.joblib`** - Naive Bayes model
7. **`model_outputs/logistic_regression.joblib`** - Logistic Regression model

### Dataset (for training):
8. **`Fake.csv`** - 23,481 fake news articles
9. **`True.csv`** - 21,417 real news articles

## 🚀 Deployment Options

### Option 1: Upload to GitHub

```bash
# Create repository
git init
git add fake_news_detector_complete.py model_requirements.txt MODEL_README.md quick_test.py
git commit -m "Initial commit: Fake News Detection System"
git remote add origin https://github.com/yourusername/fake-news-detector.git
git push -u origin main

# Add trained models (if you want to share them)
git add model_outputs/
git commit -m "Add trained models"
git push
```

### Option 2: Deploy as Web API (Flask)

Create `app.py`:

```python
from flask import Flask, request, jsonify
from fake_news_detector_complete import load_models, predict_news

app = Flask(__name__)

# Load models once at startup
vectorizer, nb_model, lr_model = load_models()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    result, conf_real, conf_fake = predict_news(text, vectorizer, nb_model)
    
    return jsonify({
        'prediction': result,
        'confidence': {
            'real': float(conf_real),
            'fake': float(conf_fake)
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Install Flask: `pip install flask`

Run: `python app.py`

Test:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists confirm Earth is flat"}'
```

### Option 3: Deploy to Cloud (Heroku)

1. Create `Procfile`:
```
web: python app.py
```

2. Create `runtime.txt`:
```
python-3.11.0
```

3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### Option 4: Package as Python Library

Create `setup.py`:

```python
from setuptools import setup, find_packages

setup(
    name='fake-news-detector',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.21.0,<2.0.0',
        'scikit-learn>=1.0.0',
        'joblib>=1.0.0',
    ],
    author='Your Name',
    description='ML-based fake news detection system',
    python_requires='>=3.8',
)
```

Install locally: `pip install -e .`

## 📤 Sharing Your Model

### Option A: Share Code Only (Small)
Upload just the Python files. Users train their own models.

**Files to share:**
- `fake_news_detector_complete.py`
- `model_requirements.txt`
- `MODEL_README.md`
- `quick_test.py`

**Size:** ~50 KB

### Option B: Share Code + Trained Models (Medium)
Include pre-trained models so users can use immediately.

**Files to share:**
- All from Option A
- `model_outputs/` folder (3 .joblib files)

**Size:** ~100-200 MB

### Option C: Share Everything (Large)
Include dataset for full reproducibility.

**Files to share:**
- All from Option B
- `Fake.csv` and `True.csv`

**Size:** ~50-100 MB (compressed)

## 🌐 Using the Model Online

### Google Colab

```python
# Upload files to Colab
from google.colab import files
uploaded = files.upload()

# Install dependencies
!pip install -r model_requirements.txt

# Run training
!python fake_news_detector_complete.py

# Test
!python quick_test.py
```

### Kaggle Notebook

1. Upload `fake_news_detector_complete.py`
2. Use Kaggle's dataset directly
3. Run training in notebook

## 🔒 Security Considerations

### For Production Deployment:

1. **Input Validation**: Limit text length
```python
MAX_TEXT_LENGTH = 10000
if len(text) > MAX_TEXT_LENGTH:
    return "Text too long"
```

2. **Rate Limiting**: Prevent abuse
```python
from flask_limiter import Limiter
limiter = Limiter(app, default_limits=["100 per hour"])
```

3. **API Authentication**: Add API keys
```python
API_KEY = "your-secret-key"
if request.headers.get('X-API-Key') != API_KEY:
    return "Unauthorized", 401
```

4. **HTTPS**: Always use SSL in production

## 📊 Monitoring

### Track Model Performance

```python
import logging

logging.basicConfig(filename='predictions.log', level=logging.INFO)

def predict_and_log(text):
    result, conf_real, conf_fake = predict_news(text, vectorizer, nb_model)
    logging.info(f"Prediction: {result}, Confidence: {conf_fake:.2f}")
    return result, conf_real, conf_fake
```

### Metrics to Monitor:
- Prediction distribution (% fake vs real)
- Confidence scores
- Response times
- Error rates

## 🔄 Updating the Model

### Retrain with New Data:

1. Add new articles to CSV files
2. Run training again:
```bash
python fake_news_detector_complete.py
```
3. Test with `quick_test.py`
4. Replace old models in production

### Version Control:

```bash
# Tag model versions
git tag -a v1.0 -m "Initial model - 94.36% accuracy"
git push origin v1.0

# For new version
git tag -a v1.1 -m "Updated model - 95.2% accuracy"
git push origin v1.1
```

## 📱 Integration Examples

### Python Script:
```python
from fake_news_detector_complete import load_models, predict_news

vectorizer, nb_model, lr_model = load_models()
result, _, _ = predict_news("Your article here", vectorizer, nb_model)
print(f"This is {result} news")
```

### Web Application:
```javascript
// Frontend (JavaScript)
fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: articleText})
})
.then(res => res.json())
.then(data => {
    console.log(`Prediction: ${data.prediction}`);
    console.log(`Confidence: ${data.confidence.fake * 100}%`);
});
```

### Command Line:
```bash
# Create CLI wrapper
echo "python -c \"from fake_news_detector_complete import *; v,m,_=load_models(); print(predict_news('$1',v,m)[0])\"" > detect.sh
chmod +x detect.sh

# Use it
./detect.sh "Your article text here"
```

## 🎯 Best Practices

1. **Always test** after deployment
2. **Monitor performance** in production
3. **Version your models** for rollback capability
4. **Document changes** in model behavior
5. **Validate inputs** before prediction
6. **Handle errors** gracefully
7. **Log predictions** for analysis
8. **Update regularly** with new data

## 📞 Support

For issues or questions:
1. Check `MODEL_README.md` for usage
2. Run `quick_test.py` to verify setup
3. Review error logs
4. Open GitHub issue (if applicable)

---

**Ready to deploy? Start with Option 1 (GitHub) for easy sharing!**
