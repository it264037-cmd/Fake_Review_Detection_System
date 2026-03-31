# ReviewGuard — ML Spam Detection System

## Project Structure
```
reviewguard/
├── backend/
│   ├── app.py              ← FastAPI REST API
│   ├── train_model.py      ← ML model trainer (run once)
│   ├── requirements.txt    ← Python dependencies
│   └── models/             ← Auto-created after training
└── frontend/
    └── index.html          ← Connected frontend
```

## Quick Start

### Step 1 — Install dependencies
```bash
cd backend
py -m pip install -r requirements.txt
```

### Step 2 — Train the ML model
```bash
py train_model.py
```

### Step 3 — Start the API server
```bash
py -m uvicorn app:app --reload --port 8000
```
Swagger docs → http://localhost:8000/docs

### Step 4 — Open the frontend
Open `frontend/index.html` in your browser.
Click "Skip" on setup screen to use demo mode, or add EmailJS keys for real OTP emails.

### Step 5 — Admin login
Email: admin@reviewguard.com  |  Password: admin123

---

## ML Detection Layers
1. Text Feature Engineering (18 features: caps ratio, repetition, punctuation density, etc.)
2. TF-IDF Vectorization (500 features, unigrams+bigrams+trigrams)
3. Ensemble ML Model (Gradient Boosting + Random Forest + Logistic Regression)
4. Behavioral Analysis (rate limiting per user and IP)
5. Duplicate Detection (cosine similarity via TF-IDF vectors)
6. Regex Pattern Rules (spam keywords, URLs, keyboard smash)

## Admin API
- GET  /api/admin/reviews?token=TOKEN&filter=all|spam|approved|pending
- GET  /api/admin/stats?token=TOKEN
- POST /api/admin/reviews/{id}/approve
- POST /api/admin/reviews/{id}/spam
- DELETE /api/admin/reviews/{id}?token=TOKEN
- POST /api/admin/bulk/approve-pending
- POST /api/admin/bulk/delete-spam
