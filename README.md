# 🎬 IMDB Movie Sentiment Analysis

A full-stack Machine Learning application that classifies movie reviews as Positive or Negative. This project compares traditional Machine Learning methods against modern Deep Learning techniques and deploys the best-performing model using a REST API.

## 🚀 Features

- **Dual Model Approach**: Compares Logistic Regression (Baseline) vs. DistilBERT (Transfer Learning).
- **High Accuracy**: The fine-tuned DistilBERT model achieves 91.46% accuracy.
- **Real-time Inference**: Fast API built with FastAPI.
- **User Interface**: Simple HTML/JS frontend for testing reviews.


## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **ML Libraries**: PyTorch, Hugging Face Transformers, Scikit-learn, Pandas, NLTK.
- **Backend**: FastAPI, Uvicorn.
- **Frontend**: HTML5, CSS3, JavaScript (Fetch API).


## 📊 Model Performance

Comparison of model accuracy on the test set:

| Model | Type | Accuracy |
|-------|------|----------|
| Logistic Regression | Baseline (Bag of Words) | 89.15% |
| DistilBERT | Transformer (Fine-tuned) | 91.46% |

## 📂 Project Structure

```
├── sentiment_model/       # Saved fine-tuned DistilBERT model files
├── Dockerfile             # Docker configuration
├── IMDB_Sentiment.ipynb   # Model training & analysis notebook
├── index.html             # User Interface
├── main.py                # FastAPI backend application
└── requirements.txt       # Python dependencies
```

