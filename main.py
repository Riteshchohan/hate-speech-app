from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

for pkg in ['stopwords', 'punkt', 'punkt_tab']:
    nltk.download(pkg, quiet=True)

app = FastAPI(title="Hate Speech Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model & vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

stop_words = set(stopwords.words("english"))
stop_words.update(["#ff", "ff", "rt"])
stemmer = SnowballStemmer("english")

def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@[\w\-]+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [stemmer.stem(w) for w in text.split() if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

class TextInput(BaseModel):
    text: str

class PredictionResult(BaseModel):
    text: str
    prediction: str
    confidence: float
    label: int

@app.get("/")
def root():
    return {"message": "Hate Speech Detection API", "status": "running"}

@app.post("/predict", response_model=PredictionResult)
def predict(input: TextInput):
    clean = preprocess(input.text)
    vec   = vectorizer.transform([clean])

    label = int(model.predict(vec)[0])

    # Confidence score
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        confidence = float(max(proba))
    elif hasattr(model, "decision_function"):
        score = model.decision_function(vec)[0]
        # Convert decision score to a 0-1 confidence
        import numpy as np
        confidence = float(1 / (1 + np.exp(-abs(score))))
    else:
        confidence = 1.0

    prediction = "Toxic / Hate Speech" if label == 1 else "Not Toxic"

    return PredictionResult(
        text=input.text,
        prediction=prediction,
        confidence=round(confidence, 4),
        label=label
    )

@app.get("/health")
def health():
    return {"status": "ok"}
