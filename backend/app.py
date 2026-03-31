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
import json
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
    import train_model as _train_model_module
    from train_model import ReviewFeatureExtractor as _ReviewFeatureExtractor
    _main.ReviewFeatureExtractor = _ReviewFeatureExtractor
except Exception:
    _train_model_module = None
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

def load_model_metadata() -> dict:
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.json")
    try:
        with open(metadata_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}
    except Exception as exc:
        print(f"[WARN] Could not load model metadata: {exc}")
        return {}


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
model_metadata = load_model_metadata()
model_lock = threading.Lock()


def prepare_review_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"\b\d{5,}\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_review_text(text: str) -> str:
    """Normalize review text for stable comparisons."""
    return prepare_review_text(text).lower()


def build_training_spam_texts() -> set:
    training_data = getattr(_train_model_module, "TRAINING_DATA", []) if _train_model_module else []
    return {
        normalize_review_text(text)
        for text, label in training_data
        if label == 1 and str(text).strip()
    }


TRAINING_SPAM_TEXTS = build_training_spam_texts()


def refresh_model_artifacts():
    global classifier, tfidf_vec, feat_extractor, model_metadata, TRAINING_SPAM_TEXTS
    classifier, tfidf_vec, feat_extractor = load_models()
    model_metadata = load_model_metadata()
    TRAINING_SPAM_TEXTS = build_training_spam_texts()


def current_model_info() -> dict:
    dataset_sources = model_metadata.get("dataset_sources") or (
        [model_metadata.get("dataset_source")] if model_metadata.get("dataset_source") else []
    )
    return {
        "trained_at": model_metadata.get("trained_at"),
        "dataset_source": model_metadata.get("dataset_source"),
        "dataset_sources": dataset_sources,
        "dataset_count": model_metadata.get("dataset_count", len(dataset_sources)),
        "samples": model_metadata.get("samples", 0),
        "spam_samples": model_metadata.get("spam_samples", 0),
        "genuine_samples": model_metadata.get("genuine_samples", 0),
        "cv_f1_mean": model_metadata.get("cv_f1_mean"),
    }


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

# Admin-configurable protection for repeated reviews from the same IP
admin_settings = {
    "ip_review_limit_enabled": True,
    "max_reviews_per_ip": 5,
    "ip_window_seconds": 600,  # 10 minutes
}
blocked_ips: dict = {}  # ip → {blocked_at, review_count, reason}

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


class AdminSettingsRequest(BaseModel):
    token: str
    ip_review_limit_enabled: bool = True
    max_reviews_per_ip: int = 5

    @field_validator('max_reviews_per_ip')
    @classmethod
    def max_reviews_in_range(cls, v):
        if v < 1 or v > 50:
            raise ValueError('Max reviews per IP must be between 1 and 50')
        return v


class RetrainModelRequest(BaseModel):
    token: str
    dataset_path: Optional[str] = None
    dataset_paths: Optional[List[str]] = None

    @field_validator('dataset_path')
    @classmethod
    def normalize_dataset_path(cls, v):
        if v is None:
            return None
        value = v.strip()
        return value or None

    @field_validator('dataset_paths')
    @classmethod
    def normalize_dataset_paths(cls, v):
        if v is None:
            return None
        items = v if isinstance(v, list) else re.split(r"[;\r\n]+", str(v))
        cleaned = [str(item).strip() for item in items if str(item).strip()]
        return cleaned or None


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
    1. ML model score
    2. Behavioral analysis
    3. Duplicate detection
    4. Rule-based pattern detection
    Returns score (0-100), flags, and moderation status.
    """
    clean_text = prepare_review_text(text)
    flags = []
    penalties = 0.0
    now = time.time()

    # ── 1. LENGTH CHECK ──────────────────────────────────────────────────────
    word_count = len(clean_text.split())
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
            eng_feats = feat_extractor.to_array(clean_text).reshape(1, -1)
            tfidf_feats = tfidf_vec.transform([clean_text]).toarray()
            X = np.hstack([eng_feats, tfidf_feats])
            ml_prob = float(classifier.predict_proba(X)[0][1])
        except Exception as e:
            print(f"ML error: {e}")

    if ml_prob >= 0.75:
        flags.append(f"ML model flagged as spam ({int(ml_prob * 100)}% confidence)")
        penalties += 30
    elif ml_prob >= 0.55:
        flags.append("ML model suspects spam content")
        penalties += 15

    # ── 3. PATTERN RULES ─────────────────────────────────────────────────────
    caps_ratio = sum(1 for c in clean_text if c.isupper()) / max(len(clean_text), 1)
    excl_count = clean_text.count('!')
    url_hits = len(re.findall(r'https?://|www\.|\.com|\.net', clean_text, re.I))
    spam_kw = SPAM_KEYWORDS.findall(clean_text)

    if caps_ratio > 0.65 and len(clean_text) > 5:
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

    words = clean_text.lower().split()
    if len(words) > 3:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.4:
            flags.append("Repetitive words detected")
            penalties += 25

    # ── 4. BEHAVIORAL ANALYSIS ───────────────────────────────────────────────
    window = 300
    user_subs = [t for t in user_submissions[user_id] if now - t < window]
    if len(user_subs) >= 3:
        flags.append(f"Too many reviews in short time ({len(user_subs) + 1} in 5 min)")
        penalties += 30
    elif len(user_subs) >= 2:
        flags.append("High submission frequency detected")
        penalties += 10

    ip_window_seconds = max(60, int(admin_settings.get("ip_window_seconds", 600)))
    max_reviews_per_ip = max(1, int(admin_settings.get("max_reviews_per_ip", 5)))
    ip_limit_enabled = bool(admin_settings.get("ip_review_limit_enabled", True))
    ip_subs = [t for t in ip_submissions[ip] if now - t < ip_window_seconds]
    if ip_limit_enabled and len(ip_subs) >= max_reviews_per_ip:
        flags.append(
            f"IP exceeded admin limit ({len(ip_subs) + 1}/{max_reviews_per_ip} reviews in {ip_window_seconds // 60} min)"
        )
        penalties += 60

    ip_user_count = len(ip_users[ip])
    if ip_user_count >= 3:
        flags.append(f"IP shared by {ip_user_count} different accounts")
        penalties += 20

    # ── 5. DUPLICATE DETECTION ───────────────────────────────────────────────
    exact_matches = 0
    near_matches = 0

    if tfidf_vec:
        try:
            new_vec = tfidf_vec.transform([clean_text]).toarray()[0]
        except Exception:
            new_vec = None
    else:
        new_vec = None

    for r in existing_reviews:
        existing_text = prepare_review_text(r.get("text", ""))
        if existing_text.lower() == clean_text.lower():
            exact_matches += 1
        elif new_vec is not None and tfidf_vec:
            try:
                ex_vec = tfidf_vec.transform([existing_text]).toarray()[0]
                sim = cosine_similarity_texts(new_vec, ex_vec)
                if sim > 0.90:
                    near_matches += 1
            except Exception:
                pass

    if exact_matches > 0:
        flags.append(f"Exact duplicate detected ({exact_matches} identical review{'s' if exact_matches > 1 else ''})")
        penalties += 35
    elif near_matches > 0:
        flags.append(f"Near-duplicate content ({near_matches} very similar review{'s' if near_matches > 1 else ''})")
        penalties += 20

    # ── 6. COMPUTE FINAL SCORE ───────────────────────────────────────────────
    raw_score = max(0, 100 - penalties)
    blended_score = int(raw_score * (1 - ml_prob * 0.4))
    final_score = max(0, min(100, blended_score))

    # ── DECISION LOGIC ───────────────────────────────────────────────────────
    normalized_text = normalize_review_text(clean_text)
    matched_training_spam = normalized_text in TRAINING_SPAM_TEXTS
    ip_already_blocked = bool(ip_limit_enabled and ip in blocked_ips)
    ip_limit_hit = bool(ip_limit_enabled and len(ip_subs) >= max_reviews_per_ip)
    high_confidence_spam = ml_prob >= 0.80
    score_too_low = final_score <= 35
    strong_rule_match = penalties >= 55 and len(flags) >= 2

    if matched_training_spam:
        flags.append("Matched a spam example from the training dataset")
    if ip_already_blocked:
        flags.append("This IP is blocked by the admin review policy")
    elif ip_limit_hit:
        flags.append(f"This IP has been blocked after more than {max_reviews_per_ip} reviews")
    if high_confidence_spam and not matched_training_spam:
        flags.append("Auto-blocked by the trained spam model")
    elif score_too_low and not (ip_already_blocked or ip_limit_hit):
        flags.append("Blocked due to very low authenticity score")

    flags = list(dict.fromkeys(flags))

    is_spam = any([
        matched_training_spam,
        ip_already_blocked,
        ip_limit_hit,
        high_confidence_spam,
        score_too_low,
        strong_rule_match,
    ])
    status = "spam" if is_spam else "pending" if (ml_prob >= 0.55 or final_score < 60 or len(flags) >= 2) else "approved"

    return {
        "score": final_score,
        "ml_prob": round(ml_prob, 3),
        "flags": flags,
        "status": status,
        "is_spam": is_spam,
        "matched_training_spam": matched_training_spam,
        "blocked_by_ip_policy": ip_already_blocked or ip_limit_hit,
        "ip_limit_hit": ip_limit_hit,
        "high_confidence_spam": high_confidence_spam,
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
    """Public endpoint: show approved and pending reviews, but hide spam."""
    visible_reviews = [
        {
            "id":      r["id"],
            "author":  r["author"],
            "text":    r["text"],
            "time":    fmt_time(r["created_at"]),
            "status":  r["status"],
        }
        for r in reviews_db
        if r["status"] in {"approved", "pending"}
    ]
    return {"reviews": visible_reviews, "total": len(visible_reviews)}


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
        now = time.time()

        # Run spam detection
        result = detect_spam(req.text, user_id, ip, reviews_db)

        if result.get("ip_limit_hit"):
            current_count = len([
                t for t in ip_submissions[ip]
                if now - t < int(admin_settings.get("ip_window_seconds", 600))
            ]) + 1
            blocked_ips[ip] = {
                "blocked_at": now,
                "review_count": current_count,
                "reason": f"Exceeded {admin_settings['max_reviews_per_ip']} reviews from the same IP",
            }

        # Update behavioral tracking
        user_submissions[user_id].append(now)
        ip_submissions[ip].append(now)
        ip_users[ip].add(user_id)
        user_review_texts[user_id].append(req.text)

        # Create review record using the moderation status returned by the detector
        review_counter["n"] += 1
        status = result["status"]
        review = {
            "id":         review_counter["n"],
            "author":     author,
            "author_id":  user_id,
            "text":       req.text,
            "time":       "just now",
            "status":     status,
            "spam":       status == "spam",
            "score":      result["score"],
            "ml_score":   result["ml_prob"],
            "flags":      result["flags"],
            "created_at": now,
            "ip":         ip,
        }
        reviews_db.insert(0, review)

    if result.get("blocked_by_ip_policy"):
        message = f"Review blocked: this IP has crossed the admin limit of {admin_settings['max_reviews_per_ip']} reviews."
    elif review["status"] == "spam":
        message = "This review was removed automatically by the trained spam model."
    elif review["status"] == "pending":
        message = "Review submitted and sent to admin review because it looks suspicious."
    else:
        message = "Review submitted and approved! It is now visible on the review page."

    return {
        "id":       review["id"],
        "status":   review["status"],
        "score":    review["score"],
        "is_spam":  review["spam"],
        "ml_score": review["ml_score"],
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


@app.get("/api/admin/settings")
async def admin_get_settings(token: str):
    require_admin(token)

    blocked_list = [
        {
            "ip": ip,
            "blocked_at": datetime.fromtimestamp(data["blocked_at"]).isoformat(),
            "blocked_time": fmt_time(data["blocked_at"]),
            "review_count": data.get("review_count", 0),
            "reason": data.get("reason", "Exceeded the IP review policy"),
        }
        for ip, data in sorted(
            blocked_ips.items(),
            key=lambda item: item[1].get("blocked_at", 0),
            reverse=True,
        )
    ]

    return {
        "ip_review_limit_enabled": admin_settings["ip_review_limit_enabled"],
        "max_reviews_per_ip": admin_settings["max_reviews_per_ip"],
        "window_minutes": admin_settings["ip_window_seconds"] // 60,
        "blocked_ips": blocked_list,
    }


@app.post("/api/admin/settings")
async def admin_update_settings(req: AdminSettingsRequest):
    require_admin(req.token)

    with db_lock:
        admin_settings["ip_review_limit_enabled"] = req.ip_review_limit_enabled
        admin_settings["max_reviews_per_ip"] = req.max_reviews_per_ip

    state = "enabled" if admin_settings["ip_review_limit_enabled"] else "disabled"
    return {
        "message": f"IP review protection {state}. Limit set to {admin_settings['max_reviews_per_ip']}.",
        "ip_review_limit_enabled": admin_settings["ip_review_limit_enabled"],
        "max_reviews_per_ip": admin_settings["max_reviews_per_ip"],
        "window_minutes": admin_settings["ip_window_seconds"] // 60,
    }


@app.post("/api/admin/ip-blocks/{ip}/remove")
async def admin_unblock_ip(ip: str, req: AdminActionRequest):
    require_admin(req.token)

    with db_lock:
        removed = blocked_ips.pop(ip, None)

    if not removed:
        raise HTTPException(status_code=404, detail="IP block not found")

    return {"message": f"IP {ip} unblocked", "ip": ip}


@app.post("/api/admin/retrain")
async def admin_retrain_model(req: RetrainModelRequest):
    require_admin(req.token)

    if _train_model_module is None:
        raise HTTPException(status_code=500, detail="Training module is unavailable on the server")

    acquired = model_lock.acquire(blocking=False)
    if not acquired:
        raise HTTPException(status_code=409, detail="Model retraining is already running")

    try:
        requested_paths = req.dataset_paths if req.dataset_paths else req.dataset_path
        start_time = time.time()
        _train_model_module.train(requested_paths)
        refresh_model_artifacts()
        elapsed = round(time.time() - start_time, 2)
        return {
            "message": "Model retrained successfully",
            "seconds": elapsed,
            "model_info": current_model_info(),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {exc}") from exc
    finally:
        model_lock.release()


# ─── HEALTH CHECK ────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status":      "ok",
        "ml_loaded":   classifier is not None,
        "users":       len(users_db),
        "reviews":     len(reviews_db),
        "timestamp":   datetime.utcnow().isoformat(),
        "model_info": current_model_info(),
    }


@app.get("/")
async def root():
    return {"message": "ReviewGuard API", "docs": "/docs", "health": "/api/health"}


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
