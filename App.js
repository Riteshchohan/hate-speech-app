import React, { useState, useRef } from 'react';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const EXAMPLES = [
  { label: "Neutral", text: "I love spending time with my family on weekends." },
  { label: "Offensive", text: "I hate those people, they are destroying everything." },
  { label: "Neutral", text: "The weather today is absolutely beautiful!" },
  { label: "Offensive", text: "Those idiots should go back to where they came from." },
];

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [history, setHistory] = useState([]);
  const textareaRef = useRef(null);

  const handlePredict = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error('API error');
      const data = await res.json();
      setResult(data);
      setHistory(prev => [{ ...data, id: Date.now() }, ...prev.slice(0, 9)]);
    } catch (e) {
      setError('Could not connect to the API. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleExample = (exText) => {
    setText(exText);
    setResult(null);
    textareaRef.current?.focus();
  };

  const confidencePct = result ? Math.round(result.confidence * 100) : 0;
  const isToxic = result?.label === 1;

  return (
    <div className="app">
      {/* Background grid */}
      <div className="bg-grid" />

      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">⚡</span>
            <span className="logo-text">HateGuard</span>
          </div>
          <div className="header-meta">
            <span className="badge">ML Model</span>
            <span className="badge">TF-IDF + SMOTE</span>
            <span className="badge">by Ritesh Chouhan</span>
          </div>
        </div>
      </header>

      <main className="main">
        <div className="hero">
          <h1 className="title">
            Hate Speech<br />
            <span className="title-accent">Detection</span>
          </h1>
          <p className="subtitle">
            NLP-powered binary classifier trained on 24,000+ tweets using
            TF-IDF, SMOTE oversampling, and ensemble model comparison.
          </p>
        </div>

        <div className="card input-card">
          <div className="card-label">INPUT TEXT</div>
          <textarea
            ref={textareaRef}
            className="textarea"
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder="Enter a tweet or any text to analyse..."
            rows={5}
            onKeyDown={e => e.ctrlKey && e.key === 'Enter' && handlePredict()}
          />
          <div className="input-footer">
            <span className="char-count">{text.length} characters</span>
            <button
              className={`btn-predict ${loading ? 'loading' : ''}`}
              onClick={handlePredict}
              disabled={loading || !text.trim()}
            >
              {loading ? (
                <span className="spinner" />
              ) : (
                <>Analyse <span className="btn-arrow">→</span></>
              )}
            </button>
          </div>
        </div>

        {/* Examples */}
        <div className="examples-section">
          <span className="examples-label">Try an example:</span>
          <div className="examples-list">
            {EXAMPLES.map((ex, i) => (
              <button
                key={i}
                className={`example-chip ${ex.label === 'Offensive' ? 'chip-red' : 'chip-green'}`}
                onClick={() => handleExample(ex.text)}
              >
                <span className="chip-dot" />
                {ex.label}
              </button>
            ))}
          </div>
        </div>

        {error && (
          <div className="card error-card">
            <span className="error-icon">⚠</span> {error}
          </div>
        )}

        {result && (
          <div className={`card result-card ${isToxic ? 'result-toxic' : 'result-clean'}`}>
            <div className="result-header">
              <div className="result-verdict">
                <div className={`verdict-badge ${isToxic ? 'verdict-toxic' : 'verdict-clean'}`}>
                  {isToxic ? '🚫 TOXIC' : '✅ SAFE'}
                </div>
                <div className="verdict-label">{result.prediction}</div>
              </div>
              <div className="confidence-block">
                <div className="confidence-number">{confidencePct}%</div>
                <div className="confidence-label">Confidence</div>
              </div>
            </div>

            <div className="confidence-bar-wrap">
              <div
                className={`confidence-bar ${isToxic ? 'bar-toxic' : 'bar-clean'}`}
                style={{ width: `${confidencePct}%` }}
              />
            </div>

            <div className="result-text-preview">
              <span className="preview-label">Analysed:</span>
              <span className="preview-text">"{result.text.slice(0, 120)}{result.text.length > 120 ? '…' : ''}"</span>
            </div>
          </div>
        )}

        {/* History */}
        {history.length > 0 && (
          <div className="history-section">
            <div className="history-header">
              <span className="history-title">Recent Analyses</span>
              <button className="clear-btn" onClick={() => setHistory([])}>Clear</button>
            </div>
            <div className="history-list">
              {history.map(item => (
                <div key={item.id} className={`history-item ${item.label === 1 ? 'hist-toxic' : 'hist-clean'}`}>
                  <span className={`hist-dot ${item.label === 1 ? 'dot-red' : 'dot-green'}`} />
                  <span className="hist-text">{item.text.slice(0, 60)}{item.text.length > 60 ? '…' : ''}</span>
                  <span className="hist-conf">{Math.round(item.confidence * 100)}%</span>
                  <span className={`hist-label ${item.label === 1 ? 'label-red' : 'label-green'}`}>
                    {item.label === 1 ? 'Toxic' : 'Safe'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tech info */}
        <div className="tech-grid">
          {[
            { icon: '🧹', label: 'Preprocessing', desc: 'Tokenisation · Stopword removal · Stemming' },
            { icon: '📊', label: 'Vectorisation', desc: 'TF-IDF · Bigrams · 10,000 features' },
            { icon: '⚖️', label: 'Imbalance', desc: 'SMOTE oversampling on minority class' },
            { icon: '🤖', label: 'Models', desc: 'LR · Naive Bayes · SVM (best by F1)' },
          ].map((t, i) => (
            <div className="tech-card" key={i}>
              <div className="tech-icon">{t.icon}</div>
              <div className="tech-label">{t.label}</div>
              <div className="tech-desc">{t.desc}</div>
            </div>
          ))}
        </div>
      </main>

      <footer className="footer">
        Built by Ritesh Chouhan · Hate Speech Detection · Python · NLTK · Scikit-learn
      </footer>
    </div>
  );
}

export default App;
