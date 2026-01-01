import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# 1. Initialize the App
app = FastAPI(
    title="IMDB Sentiment Analysis API",
    description="A simple API to classify movie reviews as Positive or Negative using DistilBERT.",
    version="1.0"
)

# Enable CORS (Cross-Origin Resource Sharing) for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development only)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],
)

# 2. Load or download the model (once on startup)
MODEL_ID = os.environ.get("MODEL_ID", "Vividvanilla/BERT-sentiment")
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "./sentiment_model"))


def load_pipeline():
    """Ensure a local model exists; download from Hugging Face if missing."""
    hf_token = os.environ.get("HF_TOKEN")

    if not MODEL_PATH.exists():
        print(f"Model directory {MODEL_PATH} not found. Downloading {MODEL_ID}...")
        tokenizer_remote = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
        model_remote = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, token=hf_token)
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        tokenizer_remote.save_pretrained(MODEL_PATH)
        model_remote.save_pretrained(MODEL_PATH)
        print("Model downloaded and saved locally.")
    else:
        print(f"Using existing model at {MODEL_PATH}.")

    tokenizer_local = AutoTokenizer.from_pretrained(MODEL_PATH)
    model_local = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return pipeline("sentiment-analysis", model=model_local, tokenizer=tokenizer_local, device=-1)


sentiment_pipeline = load_pipeline()

# 3. Define the Request Data Structure
class ReviewRequest(BaseModel):
    review: str

# 4. Define the Prediction Endpoint
@app.post("/predict")
async def predict_sentiment(request: ReviewRequest):
    if not request.review:
        raise HTTPException(status_code=400, detail="Review text cannot be empty")

    # Get raw prediction from the pipeline
    # The model returns labels like "LABEL_0" (Negative) or "LABEL_1" (Positive)
    raw_result = sentiment_pipeline(request.review)[0]
    
    # Map model labels to human-readable sentiment
    label_map = {"LABEL_0": "Negative", "LABEL_1": "Positive"}
    sentiment = label_map.get(raw_result['label'], raw_result['label'])
    
    # Convert confidence score to a percentage
    confidence = round(raw_result['score'] * 100, 2)

    return {
        "review": request.review,
        "sentiment": sentiment,
        "confidence": f"{confidence}%",
        "raw_label": raw_result['label']
    }

# 5. Root Endpoint (Health Check)
@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running. Go to /docs to test it."}