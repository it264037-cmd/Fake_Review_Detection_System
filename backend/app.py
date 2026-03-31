"""
ReviewGuard - FastAPI Backend
Spam Detection API with ML model, behavioral analysis, and admin endpoints.

Run:
  1. py -m pip install -r requirements.txt
  2. py train_model.py              (trains & saves the ML model)
  3. py -m uvicorn app:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional, List
from datetime import datetime, timedelta
import numpy as np
import joblib
import re
import os
import time
import uuid
import hashlib
from collections import defaultdict
from scipy.spatial.distance import cosine
import threading

# Backward-compatible joblib loading for locally trained custom artifacts
try:
    import __main__ as _main
    from train_model import ReviewFeatureExtractor as _ReviewFeatureExtractor
    _main.ReviewFeatureExtractor = _ReviewFeatureExtractor
except Exception:
    pass


# ─── APP SETUP ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="ReviewGuard API",
    description="Fake Review & Spam Detection System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # In production: restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)


# ─── LOAD ML ARTIFACTS ──────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def load_models():
    try:
        clf       = joblib.load(os.path.join(MODEL_DIR, "spam_classifier.pkl"))
        tfidf     = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
        extractor = joblib.load(os.path.join(MODEL_DIR, "feature_extractor.pkl"))
        print("[OK] ML models loaded successfully")
        return clf, tfidf, extractor
    except FileNotFoundError:
        print("[WARN] Models not found. Run train_model.py first.")
        return None, None, None

classifier, tfidf_vec, feat_extractor = load_models()


# ─── IN-MEMORY DATA STORE ───────────────────────────────────────────────────
# Replace with a real database (PostgreSQL, SQLite) in production
db_lock = threading.Lock()

users_db: dict = {}          # email → user record
reviews_db: list = []        # list of review dicts
review_counter = {"n": 4}    # auto-increment ID

# Behavioral tracking
ip_submissions: dict = defaultdict(list)       # ip → [timestamps]
user_submissions: dict = defaultdict(list)     # user_id → [timestamps]
user_review_texts: dict = defaultdict(list)    # user_id → [texts]
ip_users: dict = defaultdict(set)              # ip → {user_ids}

# Seed demo data
reviews_db = [
    {
        "id": 1, "author": "praveen", "author_id": "demo1",
        "text": "not good", "time": "2 min ago",
        "status": "spam", "spam": True,
        "score": 22, "ml_score": 0.78,
        "flags": ["Review is too short", "Very low authenticity score"],
        "created_at": time.time() - 120, "ip": "1.2.3.4"
    },
    {
        "id": 2, "author": "praveen", "author_id": "demo1",
        "text": "not good", "time": "8 min ago",
        "status": "spam", "spam": True,
        "score": 18, "ml_score": 0.85,
        "flags": ["Review is too short", "Duplicate review detected", "Same text posted before"],
        "created_at": time.time() - 480, "ip": "1.2.3.4"
    },
    {
        "id": 3, "author": "praveen", "author_id": "demo1",
        "text": "not good", "time": "8 min ago",
        "status": "approved", "spam": False,
        "score": 70, "ml_score": 0.32,
        "flags": [],
        "created_at": time.time() - 500, "ip": "1.2.3.5"
    },
    {
        "id": 4, "author": "praveen", "author_id": "demo1",
        "text": "leo das", "time": "8 min ago",
        "status": "approved", "spam": False,
        "score": 70, "ml_score": 0.30,
        "flags": [],
        "created_at": time.time() - 510, "ip": "1.2.3.5"
    },
]


# ─── PYDANTIC MODELS ─────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

    @field_validator('name')
    @classmethod
    def name_not_empty(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('Name must be at least 2 characters')
        return v.strip()

    @field_validator('password')
    @classmethod
    def password_length(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters')
        return v


class LoginRequest(BaseModel):
    email: str
    password: str


class ReviewRequest(BaseModel):
    text: str
    token: str

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Review text cannot be empty')
        if len(v) > 500:
            raise ValueError('Review must be under 500 characters')
        return v.strip()


class AdminActionRequest(BaseModel):
    token: str


# ─── AUTH HELPERS ────────────────────────────────────────────────────────────
ADMIN_EMAIL    = "admin@reviewguard.com"
ADMIN_PASSWORD = "admin123"

# Simple token store (use JWT in production)
active_tokens: dict = {}   # token → {user_id, is_admin, expires_at}


def create_token(user_id: str, is_admin: bool = False) -> str:
    token = str(uuid.uuid4())
    active_tokens[token] = {
        "user_id": user_id,
        "is_admin": is_admin,
        "expires_at": time.time() + 86400  # 24h
    }
    return token


def verify_token(token: str) -> Optional[dict]:
    if not token or token not in active_tokens:
        return None
    session = active_tokens[token]
    if time.time() > session["expires_at"]:
        del active_tokens[token]
        return None
    return session


def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def fmt_time(ts: float) -> str:
    diff = time.time() - ts
    if diff < 60:        return "just now"
    if diff < 3600:      return f"{int(diff//60)} min ago"
    if diff < 86400:     return f"{int(diff//3600)} hr ago"
    return f"{int(diff//86400)} days ago"


# ─── ML SPAM DETECTION ──────────────────────────────────────────────────────
SPAM_KEYWORDS = re.compile(
    r'buy now|click here|limited offer|free gift|visit.*site|'
    r'check out.*\.com|discount code|sponsored review|paid.*review|'
    r'make.*\$\d+|earn money|giveaway|bit\.ly|tinyurl',
    re.IGNORECASE
)


def cosine_similarity_texts(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two TF-IDF vectors."""
    try:
        if np.all(v1 == 0) or np.all(v2 == 0):
            return 0.0
        return float(1.0 - cosine(v1, v2))
    except Exception:
        return 0.0


def detect_spam(
    text: str,
    user_id: str,
    ip: str,
    existing_reviews: list
) -> dict:
    """
    Multi-layer spam detection:
    1. ML model score (gradient boosting ensemble)
    2. Behavioral analysis (frequency, rate limiting)
    3. Duplicate detection (cosine similarity)
    4. Rule-based pattern detection
    Returns a result dict with score (0-100), flags, and spam decision.
    """
    flags = []
    penalties = 0.0
    now = time.time()

    # ── 1. LENGTH CHECK ──────────────────────────────────────────────────────
    word_count = len(text.split())
    if word_count < 3:
        flags.append("Review is too short (under 3 words)")
        penalties += 40
    elif word_count < 6:
        flags.append("Review is very short")
        penalties += 15

    # ── 2. ML MODEL ──────────────────────────────────────────────────────────
    ml_prob = 0.5
    if classifier and tfidf_vec and feat_extractor:
        try:
            eng_feats = feat_extractor.to_array(text).reshape(1, -1)
            tfidf_feats = tfidf_vec.transform([text]).toarray()
            X = np.hstack([eng_feats, tfidf_feats])
            ml_prob = float(classifier.predict_proba(X)[0][1])  # P(spam)
        except Exception as e:
            print(f"ML error: {e}")

    if ml_prob >= 0.75:
        flags.append(f"ML model flagged as spam ({int(ml_prob*100)}% confidence)")
        penalties += 30
    elif ml_prob >= 0.55:
        flags.append("ML model suspects spam content")
        penalties += 15

    # ── 3. PATTERN RULES ─────────────────────────────────────────────────────
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    excl_count = text.count('!')
    url_hits   = len(re.findall(r'https?://|www\.|\.com|\.net', text, re.I))
    spam_kw    = SPAM_KEYWORDS.findall(text)

    if caps_ratio > 0.65 and len(text) > 5:
        flags.append("Excessive capitalization (shouting)")
        penalties += 20
    if excl_count >= 4:
        flags.append(f"Excessive exclamation marks ({excl_count})")
        penalties += 15
    if url_hits > 0:
        flags.append("Contains URL/external link")
        penalties += 25
    if spam_kw:
        flags.append(f"Spam keywords detected: {', '.join(set(spam_kw[:3]))}")
        penalties += 35

    # Repeated word detection
    words = text.lower().split()
    if len(words) > 3:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.4:
            flags.append("Repetitive words detected")
            penalties += 25

    # ── 4. BEHAVIORAL ANALYSIS ───────────────────────────────────────────────
    # Rate limit: max 3 reviews per 5 minutes per user
    window = 300  # 5 min
    user_subs = [t for t in user_submissions[user_id] if now - t < window]
    if len(user_subs) >= 3:
        flags.append(f"Too many reviews in short time ({len(user_subs)+1} in 5 min)")
        penalties += 30
    elif len(user_subs) >= 2:
        flags.append("High submission frequency detected")
        penalties += 10

    # IP-based rate limiting: max 5 reviews per 10 minutes per IP
    ip_subs = [t for t in ip_submissions[ip] if now - t < 600]
    if len(ip_subs) >= 5:
        flags.append(f"IP address submitting too many reviews")
        penalties += 25

    # Multiple accounts from same IP
    ip_user_count = len(ip_users[ip])
    if ip_user_count >= 3:
        flags.append(f"IP shared by {ip_user_count} different accounts")
        penalties += 20

    # ── 5. DUPLICATE DETECTION ───────────────────────────────────────────────
    exact_matches = 0
    near_matches  = 0
    
    if tfidf_vec:
        try:
            new_vec = tfidf_vec.transform([text]).toarray()[0]
        except Exception:
            new_vec = None
    else:
        new_vec = None

    for r in existing_reviews:
        existing_text = r.get("text", "")
        # Exact match
        if existing_text.strip().lower() == text.strip().lower():
            exact_matches += 1
        # Near-duplicate via cosine similarity
        elif new_vec is not None and tfidf_vec:
            try:
                ex_vec = tfidf_vec.transform([existing_text]).toarray()[0]
                sim = cosine_similarity_texts(new_vec, ex_vec)
                if sim > 0.90:
                    near_matches += 1
            except Exception:
                pass

    if exact_matches > 0:
        flags.append(f"Exact duplicate detected ({exact_matches} identical review{'s' if exact_matches>1 else ''})")
        penalties += 35
    elif near_matches > 0:
        flags.append(f"Near-duplicate content ({near_matches} very similar review{'s' if near_matches>1 else ''})")
        penalties += 20

    # ── 6. COMPUTE FINAL SCORE ───────────────────────────────────────────────
    # Score = 100 (genuine) minus penalties, floored at 0
    raw_score = max(0, 100 - penalties)
    # Blend with ML: weight towards lower score if ML says spam
    blended_score = int(raw_score * (1 - ml_prob * 0.4))
    final_score   = max(0, min(100, blended_score))
    is_spam       = final_score < 50 or penalties >= 40

    return {
        "score":    final_score,
        "ml_prob":  round(ml_prob, 3),
        "flags":    flags,
        "is_spam":  is_spam,
        "penalties": penalties,
    }


# ─── ROUTES: AUTH ────────────────────────────────────────────────────────────
@app.post("/api/auth/register", status_code=201)
async def register(req: RegisterRequest):
    email = req.email.lower().strip()
    if email in users_db:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_id = str(uuid.uuid4())
    users_db[email] = {
        "id":       user_id,
        "name":     req.name,
        "email":    email,
        "password": hash_password(req.password),
        "is_admin": False,
        "created_at": time.time()
    }
    token = create_token(user_id)
    return {
        "token":    token,
        "user_id":  user_id,
        "name":     req.name,
        "email":    email,
        "is_admin": False
    }


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    email = req.email.lower().strip()
    
    # Admin check
    if email == ADMIN_EMAIL and req.password == ADMIN_PASSWORD:
        token = create_token("admin", is_admin=True)
        return {"token": token, "name": "Admin", "email": email, "is_admin": True, "user_id": "admin"}
    
    user = users_db.get(email)
    if not user or user["password"] != hash_password(req.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    token = create_token(user["id"])
    return {
        "token":    token,
        "user_id":  user["id"],
        "name":     user["name"],
        "email":    email,
        "is_admin": False
    }


@app.post("/api/auth/logout")
async def logout(req: AdminActionRequest):
    if req.token in active_tokens:
        del active_tokens[req.token]
    return {"message": "Logged out"}


# ─── ROUTES: REVIEWS ─────────────────────────────────────────────────────────
@app.get("/api/reviews")
async def get_reviews():
    """Public endpoint: only approved reviews visible to users."""
    approved = [
        {
            "id":      r["id"],
            "author":  r["author"],
            "text":    r["text"],
            "time":    fmt_time(r["created_at"]),
            "status":  r["status"],
        }
        for r in reviews_db
        if r["status"] == "approved"
    ]
    return {"reviews": approved, "total": len(approved)}


@app.post("/api/reviews", status_code=201)
async def submit_review(req: ReviewRequest, request: Request):
    """Submit a review — runs full ML + behavioral spam detection."""
    session = verify_token(req.token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    user_id = session["user_id"]
    ip      = get_client_ip(request)
    
    # Find user name
    author  = "anonymous"
    for u in users_db.values():
        if u["id"] == user_id:
            author = u["name"]
            break
    if user_id == "admin":
        author = "Admin"
    
    with db_lock:
        # Run spam detection
        result = detect_spam(req.text, user_id, ip, reviews_db)
        
        # Update behavioral tracking
        now = time.time()
        user_submissions[user_id].append(now)
        ip_submissions[ip].append(now)
        ip_users[ip].add(user_id)
        user_review_texts[user_id].append(req.text)
        
        # Create review record — status driven by spam detection
        review_counter["n"] += 1
        status = "spam" if result["is_spam"] else "approved"
        review = {
            "id":         review_counter["n"],
            "author":     author,
            "author_id":  user_id,
            "text":       req.text,
            "time":       "just now",
            "status":     status,        # Correctly set by ML + behavioral analysis
            "spam":       result["is_spam"],
            "score":      result["score"],
            "ml_score":   result["ml_prob"],
            "flags":      result["flags"],
            "created_at": now,
            "ip":         ip,
        }
        reviews_db.insert(0, review)
    
    if result["is_spam"]:
        message = "Your review was flagged as spam and will not be shown publicly."
    else:
        message = "Review submitted and approved!"

    return {
        "id":       review["id"],
        "status":   review["status"],
        "score":    review["score"],
        "is_spam":  review["spam"],
        "flags":    review["flags"],
        "message":  message
    }


# ─── ROUTES: ADMIN ───────────────────────────────────────────────────────────
def require_admin(token: str):
    session = verify_token(token)
    if not session or not session["is_admin"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    return session


@app.get("/api/admin/reviews")
async def admin_get_reviews(token: str, filter: str = "all"):
    require_admin(token)
    
    filtered = reviews_db
    if filter == "spam":
        filtered = [r for r in reviews_db if r["status"] == "spam"]
    elif filter == "approved":
        filtered = [r for r in reviews_db if r["status"] == "approved"]
    elif filter == "pending":
        filtered = [r for r in reviews_db if r["status"] == "pending"]
    
    return {
        "reviews": [
            {**r, "time": fmt_time(r["created_at"])}
            for r in filtered
        ],
        "total": len(filtered)
    }


@app.get("/api/admin/stats")
async def admin_stats(token: str):
    require_admin(token)
    
    n       = len(reviews_db)
    spam    = sum(1 for r in reviews_db if r["status"] == "spam")
    approved= sum(1 for r in reviews_db if r["status"] == "approved")
    pending = sum(1 for r in reviews_db if r["status"] == "pending")
    avg_score = int(sum(r["score"] for r in reviews_db) / n) if n else 0
    
    # Trend data (last 7 days, grouped by day)
    now = time.time()
    trend = []
    for i in range(6, -1, -1):
        day_start = now - (i+1) * 86400
        day_end   = now - i * 86400
        day_reviews = [r for r in reviews_db if day_start <= r["created_at"] < day_end]
        date_label = datetime.fromtimestamp(day_end).strftime("%b %d")
        trend.append({
            "date":     date_label,
            "total":    len(day_reviews),
            "spam":     sum(1 for r in day_reviews if r["status"] == "spam"),
            "approved": sum(1 for r in day_reviews if r["status"] == "approved"),
        })
    
    # Score distribution
    score_buckets = {"0-20": 0, "21-40": 0, "41-60": 0, "61-80": 0, "81-100": 0}
    for r in reviews_db:
        s = r["score"]
        if s <= 20:      score_buckets["0-20"] += 1
        elif s <= 40:    score_buckets["21-40"] += 1
        elif s <= 60:    score_buckets["41-60"] += 1
        elif s <= 80:    score_buckets["61-80"] += 1
        else:            score_buckets["81-100"] += 1
    
    return {
        "total":     n,
        "approved":  approved,
        "spam":      spam,
        "pending":   pending,
        "avg_score": avg_score,
        "spam_rate": round(spam / n * 100, 1) if n else 0,
        "trend":     trend,
        "score_distribution": score_buckets,
    }


@app.post("/api/admin/reviews/{review_id}/approve")
async def admin_approve(review_id: int, req: AdminActionRequest):
    require_admin(req.token)
    
    with db_lock:
        for r in reviews_db:
            if r["id"] == review_id:
                r["status"] = "approved"
                r["spam"]   = False
                r["score"]  = max(r["score"], 70)
                r["flags"]  = []
                return {"message": "Approved", "id": review_id}
    
    raise HTTPException(status_code=404, detail="Review not found")


@app.post("/api/admin/reviews/{review_id}/spam")
async def admin_mark_spam(review_id: int, req: AdminActionRequest):
    require_admin(req.token)
    
    with db_lock:
        for r in reviews_db:
            if r["id"] == review_id:
                r["status"] = "spam"
                r["spam"]   = True
                r["score"]  = min(r["score"], 35)
                return {"message": "Marked as spam", "id": review_id}
    
    raise HTTPException(status_code=404, detail="Review not found")


@app.delete("/api/admin/reviews/{review_id}")
async def admin_delete(review_id: int, token: str):
    require_admin(token)
    
    with db_lock:
        global reviews_db
        before = len(reviews_db)
        reviews_db = [r for r in reviews_db if r["id"] != review_id]
        if len(reviews_db) == before:
            raise HTTPException(status_code=404, detail="Review not found")
    
    return {"message": "Deleted", "id": review_id}


@app.post("/api/admin/bulk/approve-pending")
async def bulk_approve(req: AdminActionRequest):
    require_admin(req.token)
    
    with db_lock:
        count = 0
        for r in reviews_db:
            if r["status"] == "pending":
                r["status"] = "approved"
                r["spam"]   = False
                r["score"]  = max(r["score"], 70)
                r["flags"]  = []
                count += 1
    
    return {"message": f"Approved {count} pending reviews", "count": count}


@app.post("/api/admin/bulk/delete-spam")
async def bulk_delete_spam(req: AdminActionRequest):
    require_admin(req.token)
    
    with db_lock:
        global reviews_db
        spam_count = sum(1 for r in reviews_db if r["status"] == "spam")
        reviews_db = [r for r in reviews_db if r["status"] != "spam"]
    
    return {"message": f"Deleted {spam_count} spam reviews", "count": spam_count}


# ─── HEALTH CHECK ────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status":      "ok",
        "ml_loaded":   classifier is not None,
        "users":       len(users_db),
        "reviews":     len(reviews_db),
        "timestamp":   datetime.utcnow().isoformat()
    }


@app.get("/")
async def root():
    return {"message": "ReviewGuard API", "docs": "/docs", "health": "/api/health"}


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
