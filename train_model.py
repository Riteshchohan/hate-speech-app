"""
Hate Speech Detection — Final Training Script
Ritesh Chouhan
Dataset: hatespeech.csv (24,783 tweets)
"""

import pandas as pd
import numpy as np
import pickle
import re
import html
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer as CV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE

# ── 1. LOAD ───────────────────────────────────
print("📦 Loading dataset...")
df = pd.read_csv("hatespeech.csv")
df = df.drop(columns=['Unnamed: 0'], errors='ignore')
print(f"Raw shape: {df.shape}")
print(f"Class distribution:\n{df['class'].value_counts()}")

# ── 2. CLEAN DATA ─────────────────────────────
print("\n🧹 Cleaning data...")

df['tweet'] = df['tweet'].apply(lambda x: html.unescape(str(x)))

def clean_tweet(text):
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@[\w\-]+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\bRT\b', '', text)
    text = re.sub(r'&amp;|&lt;|&gt;|&quot;|&#\d+;', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

df['clean_tweet'] = df['tweet'].apply(clean_tweet)

before = len(df)
df = df[df['clean_tweet'].apply(lambda x: len(x.split()) >= 2)]
print(f"Dropped {before - len(df)} too-short tweets")

before = len(df)
df = df.drop_duplicates(subset=['clean_tweet'])
print(f"Dropped {before - len(df)} duplicate tweets")

df['label'] = df['class'].apply(lambda x: 1 if x == 0 else 0)
print(f"\nFinal: {len(df)} rows | Labels: {dict(df['label'].value_counts())}")

# ── 3. EDA ────────────────────────────────────
print("\n📊 Generating EDA charts...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Hate Speech Dataset — EDA (Ritesh Chouhan)", fontsize=13, fontweight='bold')

label_counts = df['label'].value_counts().sort_index()
axes[0].bar(['Not Toxic', 'Toxic'], label_counts.values,
            color=['#00e396', '#ff4560'], edgecolor='black', width=0.5)
axes[0].set_title("Binary Class Distribution")
axes[0].set_ylabel("Count")
for i, v in enumerate(label_counts.values):
    axes[0].text(i, v + 60, str(v), ha='center', fontweight='bold')

orig_counts = df['class'].value_counts().sort_index()
orig_labels = {0: 'Hate Speech', 1: 'Offensive', 2: 'Neither'}
axes[1].bar([orig_labels[i] for i in orig_counts.index], orig_counts.values,
            color=['#ff4560', '#ff9800', '#4caf50'], edgecolor='black', width=0.5)
axes[1].set_title("Original 3-Class Distribution")
axes[1].set_ylabel("Count")
for bar, v in zip(axes[1].patches, orig_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 40, str(v), ha='center', fontweight='bold')

df['word_count'] = df['clean_tweet'].apply(lambda x: len(x.split()))
for lbl, color in [(0, '#00e396'), (1, '#ff4560')]:
    axes[2].hist(df[df['label'] == lbl]['word_count'], bins=30, alpha=0.65, color=color, label=f'{"Not Toxic" if lbl==0 else "Toxic"}')
axes[2].set_title("Word Count by Class")
axes[2].set_xlabel("Words")
axes[2].legend()

plt.tight_layout()
plt.savefig("eda_distribution.png", dpi=150)
plt.close()
print("✅ eda_distribution.png saved")

toxic_tweets = df[df['label'] == 1]['clean_tweet'].tolist()
cv_tmp = CV(ngram_range=(1, 2), max_features=15, stop_words='english')
cv_tmp.fit(toxic_tweets)
counts = cv_tmp.transform(toxic_tweets).sum(axis=0).A1
terms  = cv_tmp.get_feature_names_out()
ngram_df = pd.DataFrame({'term': terms, 'count': counts}).sort_values('count', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(ngram_df['term'], ngram_df['count'], color='#ff4560', edgecolor='black')
plt.title("Top N-grams in Toxic / Hate Speech Tweets", fontweight='bold')
plt.xlabel("Frequency")
plt.tight_layout()
plt.savefig("eda_ngrams.png", dpi=150)
plt.close()
print("✅ eda_ngrams.png saved")

# ── 4. TF-IDF ─────────────────────────────────
print("\n📐 TF-IDF vectorisation...")
X = df['clean_tweet']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)
print(f"Train: {X_train_tfidf.shape} | Test: {X_test_tfidf.shape}")

# ── 5. SMOTE ──────────────────────────────────
print("\n⚖️  SMOTE oversampling...")
X_train_sm, y_train_sm = SMOTE(random_state=42).fit_resample(X_train_tfidf, y_train)
print(f"After SMOTE: {Counter(y_train_sm)}")

# ── 6. TRAIN MODELS ───────────────────────────
print("\n🤖 Training & comparing models...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Naive Bayes":         MultinomialNB(alpha=0.1),
    "SVM (LinearSVC)":     LinearSVC(C=1.0, max_iter=2000, random_state=42),
}

results, trained_models = {}, {}

for name, model in models.items():
    print(f"\n  ⏳ {name}...")
    if name == "Naive Bayes":
        Xtr = X_train_sm.copy(); Xtr[Xtr < 0] = 0
        model.fit(Xtr, y_train_sm)
    else:
        model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test_tfidf)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[name] = f1
    trained_models[name] = (model, y_pred)
    print(f"  ✅ F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Not Toxic', 'Toxic']))

# ── 7. BEST MODEL ─────────────────────────────
best_name = max(results, key=results.get)
best_model, best_preds = trained_models[best_name]
print(f"\n🏆 Best: {best_name} | F1 = {results[best_name]:.4f}")

plt.figure(figsize=(8, 5))
colors = ['#2196F3', '#FF9800', '#9C27B0']
bars = plt.bar(results.keys(), results.values(), color=colors, edgecolor='black', width=0.5)
plt.title("Model Comparison — Weighted F1 Score", fontweight='bold')
plt.ylabel("F1 Score"); plt.ylim(0.7, 1.0)
for bar, val in zip(bars, results.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f"{val:.4f}", ha='center', fontweight='bold')
plt.tight_layout(); plt.savefig("model_comparison.png", dpi=150); plt.close()
print("✅ model_comparison.png saved")

cm = confusion_matrix(y_test, best_preds)
ConfusionMatrixDisplay(cm, display_labels=['Not Toxic', 'Toxic']).plot(cmap='Blues')
plt.title(f"Confusion Matrix — {best_name}", fontweight='bold')
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()
print("✅ confusion_matrix.png saved")

# ── 8. SAVE ───────────────────────────────────
with open("model.pkl", "wb") as f: pickle.dump(best_model, f)
with open("vectorizer.pkl", "wb") as f: pickle.dump(tfidf, f)
print("\n✅ model.pkl saved")
print("✅ vectorizer.pkl saved")
print(f"\n🎉 Done! Best: {best_name} | F1 = {results[best_name]:.4f}")
