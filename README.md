# 🛡️ HateGuard — Hate Speech Detection

Built by **Ritesh Chouhan** · Python · NLTK · Scikit-learn · TF-IDF · SMOTE · React · FastAPI

---

## 📁 Project Structure

```
hate-speech-app/
├── backend/
│   ├── train_model.py      ← Run this first to generate model.pkl
│   ├── main.py             ← FastAPI server
│   ├── requirements.txt
│   ├── render.yaml         ← Deploy backend to Render
│   ├── model.pkl           ← Generated after training
│   └── vectorizer.pkl      ← Generated after training
└── frontend/
    ├── src/
    │   ├── App.js
    │   ├── App.css
    │   └── index.js
    ├── public/index.html
    ├── package.json
    └── vercel.json         ← Deploy frontend to Vercel
```

---

## 🚀 Step-by-Step Setup

### Step 1 — Train the Model

1. Download dataset from Kaggle:
   https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset
   Save as `labeled_data.csv` in the `backend/` folder.

2. Install Python dependencies:
   ```bash
   cd backend
   pip install pandas scikit-learn imbalanced-learn nltk matplotlib seaborn
   ```

3. Run training:
   ```bash
   python train_model.py
   ```
   This creates `model.pkl` and `vectorizer.pkl`.

---

### Step 2 — Test Backend Locally

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Visit: http://localhost:8000/docs (Swagger UI to test the API)

Test with curl:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I hate those people"}'
```

---

### Step 3 — Test Frontend Locally

```bash
cd frontend
npm install
npm start
```

Visit: http://localhost:3000

---

### Step 4 — Push to GitHub

```bash
# In the root hate-speech-app/ folder:
git init
git add .
git commit -m "Initial commit: Hate Speech Detection App"
git branch -M main

# Create a new repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/hate-speech-detector.git
git push -u origin main
```

---

### Step 5 — Deploy Backend to Render (Free)

1. Go to https://render.com and sign up
2. Click **New → Web Service**
3. Connect your GitHub repo
4. Set **Root Directory** to `backend`
5. Set **Build Command**: `pip install -r requirements.txt`
6. Set **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
7. Click **Deploy**
8. Copy your Render URL (e.g. `https://hate-speech-api.onrender.com`)

⚠️ You must also upload `model.pkl` and `vectorizer.pkl` to your GitHub repo (they are needed at runtime).

---

### Step 6 — Deploy Frontend to Vercel

1. Go to https://vercel.com and sign up with GitHub
2. Click **Add New Project** → import your GitHub repo
3. Set **Root Directory** to `frontend`
4. Add Environment Variable:
   - Key: `REACT_APP_API_URL`
   - Value: `https://your-render-url.onrender.com`
5. Click **Deploy**

Your app is now live! 🎉

---

## 🔬 ML Details

| Component | Details |
|---|---|
| Dataset | Twitter Hate Speech (24,000+ tweets) |
| Preprocessing | Tokenisation, stopword removal, stemming |
| Vectorisation | TF-IDF (10,000 features, bigrams) |
| Imbalance | SMOTE oversampling |
| Models | Logistic Regression, Naive Bayes, SVM |
| Selection | Best by weighted F1-score |
| Labels | Binary: 0 = Not Toxic, 1 = Toxic |
