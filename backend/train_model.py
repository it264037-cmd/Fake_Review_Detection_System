"""
ReviewGuard - ML Model Trainer
Trains the spam detection model using a CSV dataset when available.

Run:
  py train_model.py
  py train_model.py "C:\\Users\\D.praveen\\Downloads\\train_8000.csv"
  py train_model.py "file1.csv" "file2.csv" "file3.csv"
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

FALLBACK_TRAINING_DATA = [
    ("This phone is reliable and the battery life lasts all day.", 0),
    ("Good value for the price and the setup was very easy.", 0),
    ("Camera quality is strong and performance feels smooth.", 0),
    ("The design looks premium and the display is very sharp.", 0),
    ("BUY NOW AMAZING DEAL CLICK HERE LIMITED OFFER!!!", 1),
    ("Free gift giveaway! Visit our site right now!", 1),
    ("best best best best amazing amazing amazing", 1),
    ("I got paid to write this review please buy it", 1),
]

DEFAULT_DATASET_CANDIDATES: list[Path] = []
if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        if str(arg).strip():
            DEFAULT_DATASET_CANDIDATES.append(Path(arg).expanduser())

dataset_env = os.environ.get("REVIEWGUARD_DATASET_PATH")
if dataset_env:
    for raw in re.split(r"[;\r\n]+", dataset_env):
        raw = raw.strip()
        if raw:
            DEFAULT_DATASET_CANDIDATES.append(Path(raw).expanduser())

DEFAULT_DATASET_CANDIDATES.extend([
    Path(r"C:\Users\D.praveen\Downloads\dataset_3.csv"),
    Path(r"C:\Users\D.praveen\Downloads\reviews_10000 (2).csv"),
    Path(r"C:\Users\D.praveen\Downloads\reviews_5000 (1).csv"),
    Path(r"C:\Users\D.praveen\Downloads\fake_spam_5000 (1).csv"),
    Path(r"C:\Users\D.praveen\Downloads\dataset_3 (1).csv"),
    Path(r"C:\Users\D.praveen\Downloads\dataset_2 (1).csv"),
    Path(r"C:\Users\D.praveen\Downloads\dataset_1 (1).csv"),
    Path(r"C:\Users\D.praveen\Downloads\dataset_10.csv"),
    Path(r"C:\Users\D.praveen\Downloads\dataset_9.csv"),
    Path(r"C:\Users\D.praveen\Downloads\dataset_9 (1).csv"),
    Path(r"C:\Users\D.praveen\Downloads\dataset_8.csv"),
    Path(r"C:\Users\D.praveen\Downloads\dataset_7.csv"),
    Path(r"C:\Users\D.praveen\Downloads\dataset_6.csv"),
    Path(r"C:\Users\D.praveen\Downloads\dataset_5.csv"),
    Path(r"C:\Users\D.praveen\Downloads\dataset_4 (1).csv"),
    Path(r"C:\Users\D.praveen\Downloads\dataset_4.csv"),
    BASE_DIR / "train_8000.csv",
    BASE_DIR / "data" / "train_8000.csv",
])


def clean_review_text(text: str) -> str:
    """Normalize review text so training and inference use the same format."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\b\d{5,}\b", " ", text)  # strip synthetic numeric suffixes
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_dataset_inputs(value) -> list[Path]:
    if value is None:
        return []

    items = value if isinstance(value, (list, tuple, set)) else [value]
    results: list[Path] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, Path):
            results.append(item.expanduser())
            continue

        text = str(item).strip()
        if not text:
            continue

        parts = re.split(r"[;\r\n]+", text) if any(sep in text for sep in (";", "\n", "\r")) else [text]
        for part in parts:
            part = part.strip()
            if part:
                results.append(Path(part).expanduser())
    return results


def resolve_dataset_paths(path_input=None) -> list[Path]:
    candidates = parse_dataset_inputs(path_input)
    if not candidates:
        candidates = list(DEFAULT_DATASET_CANDIDATES)

    resolved: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or not str(candidate).strip():
            continue
        try:
            if candidate.exists() and candidate.is_file():
                full = str(candidate.resolve())
                if full not in seen:
                    resolved.append(candidate.resolve())
                    seen.add(full)
        except OSError:
            continue
    return resolved


def summarize_dataset_sources(sources: list[str]) -> str:
    if not sources:
        return "built-in fallback dataset"
    if len(sources) == 1:
        return sources[0]
    names = [Path(src).name for src in sources[:3]]
    extra = f", +{len(sources) - 3} more" if len(sources) > 3 else ""
    return f"{len(sources)} CSV files ({', '.join(names)}{extra})"


def normalize_label(raw_label) -> int | None:
    label = str(raw_label).strip().lower()
    if label in {"0", "real", "genuine", "ham", "approved", "legit"}:
        return 0
    if label in {"1", "spam", "fake", "fraud", "bot", "blocked"}:
        return 1
    return None


def load_training_rows(csv_path=None) -> tuple[list[tuple[str, int]], str, list[str]]:
    dataset_paths = resolve_dataset_paths(csv_path)
    if not dataset_paths:
        print("[WARN] CSV dataset not found. Falling back to built-in sample data.")
        return FALLBACK_TRAINING_DATA, "built-in fallback dataset", []

    all_rows: list[tuple[str, int]] = []
    sources: list[str] = []
    total_skipped = 0

    for dataset_path in dataset_paths:
        df = pd.read_csv(dataset_path)
        column_map = {str(col).strip().lower(): col for col in df.columns}

        text_col = next(
            (column_map[name] for name in ("review", "text", "review_text", "content", "comment") if name in column_map),
            None,
        )
        label_col = next(
            (column_map[name] for name in ("label", "target", "class", "category", "spam", "is_spam") if name in column_map),
            None,
        )

        if not text_col or not label_col:
            raise ValueError(
                f"CSV must contain review/text and label columns. Found columns: {list(df.columns)}"
            )

        file_rows: list[tuple[str, int]] = []
        skipped = 0
        for _, row in df[[text_col, label_col]].iterrows():
            text = clean_review_text(row[text_col])
            label = normalize_label(row[label_col])
            if text and label is not None:
                file_rows.append((text, label))
            else:
                skipped += 1

        if not file_rows:
            raise ValueError(f"No valid training rows were found in: {dataset_path}")

        total_skipped += skipped
        all_rows.extend(file_rows)
        sources.append(str(dataset_path))
        print(f"[OK] Loaded {len(file_rows)} rows from CSV dataset: {dataset_path}")

    if total_skipped:
        print(f"[INFO] Skipped {total_skipped} invalid rows during CSV cleanup.")

    return all_rows, summarize_dataset_sources(sources), sources


TRAINING_DATA, DATASET_SOURCE, DATASET_SOURCES = load_training_rows()


# ─── FEATURE ENGINEERING ────────────────────────────────────────────────────
class ReviewFeatureExtractor:
    """Extract engineered features from review text for ML classification."""

    SPAM_PATTERNS = [
        r'buy now', r'click here', r'limited offer', r'free gift',
        r'visit.*website', r'check out.*\.com', r'discount code',
        r'use code', r'sponsored', r'paid.*review', r'make.*\$\d+',
        r'earn money', r'work from home', r'giveaway',
    ]

    def __init__(self):
        self.spam_pattern = re.compile('|'.join(self.SPAM_PATTERNS), re.IGNORECASE)

    def extract(self, text: str) -> dict:
        text = clean_review_text(text)
        words = text.split()
        word_count = len(words)
        char_count = len(text)

        unique_words = set(w.lower() for w in words)
        unique_ratio = len(unique_words) / max(word_count, 1)

        word_freq = {}
        for w in words:
            word_freq[w.lower()] = word_freq.get(w.lower(), 0) + 1
        max_word_freq = max(word_freq.values(), default=0)
        repetition_score = max_word_freq / max(word_count, 1)

        caps_count = sum(1 for c in text if c.isupper())
        caps_ratio = caps_count / max(char_count, 1)

        punct_count = sum(1 for c in text if c in '!?.,;:')
        punct_ratio = punct_count / max(char_count, 1)

        excl_count = text.count('!')
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / max(char_count, 1)
        url_count = len(re.findall(r'https?://|www\.|\.com|\.net|\.org', text, re.IGNORECASE))
        spam_pattern_hits = len(self.spam_pattern.findall(text))

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)

        digit_count = sum(1 for c in text if c.isdigit())
        digit_ratio = digit_count / max(char_count, 1)
        lex_diversity = len(unique_words) / max(word_count, 1)

        consecutive_same = 0
        for i in range(1, len(text)):
            if text[i] == text[i - 1]:
                consecutive_same += 1
        consecutive_ratio = consecutive_same / max(char_count, 1)

        return {
            'word_count': word_count,
            'char_count': char_count,
            'unique_ratio': unique_ratio,
            'repetition_score': repetition_score,
            'caps_ratio': caps_ratio,
            'punct_ratio': punct_ratio,
            'excl_count': min(excl_count, 20),
            'special_ratio': special_ratio,
            'url_count': min(url_count, 10),
            'spam_pattern_hits': spam_pattern_hits,
            'avg_sentence_len': avg_sentence_len,
            'digit_ratio': digit_ratio,
            'lex_diversity': lex_diversity,
            'consecutive_ratio': consecutive_ratio,
            'is_very_short': int(word_count < 4),
            'is_all_caps': int(caps_ratio > 0.7 and char_count > 5),
            'has_url': int(url_count > 0),
            'has_spam_pattern': int(spam_pattern_hits > 0),
        }

    def to_array(self, text: str) -> np.ndarray:
        feats = self.extract(text)
        return np.array(list(feats.values()), dtype=np.float32)


# Make the custom extractor loadable from a stable module path when pickled.
sys.modules.setdefault("train_model", sys.modules[__name__])
ReviewFeatureExtractor.__module__ = "train_model"


# ─── TRAIN MODEL ────────────────────────────────────────────────────────────
def train(csv_path=None) -> None:
    global TRAINING_DATA, DATASET_SOURCE, DATASET_SOURCES

    if csv_path is not None:
        TRAINING_DATA, DATASET_SOURCE, DATASET_SOURCES = load_training_rows(csv_path)

    print("=" * 60)
    print("ReviewGuard ML Model Trainer")
    print("=" * 60)
    print(f"Dataset source: {DATASET_SOURCE}")

    extractor = ReviewFeatureExtractor()
    texts = [text for text, _ in TRAINING_DATA]
    labels = [label for _, label in TRAINING_DATA]

    X_eng = np.array([extractor.to_array(t) for t in texts])

    tfidf = TfidfVectorizer(
        max_features=800,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        dtype=np.float32,
    )
    X_tfidf = tfidf.fit_transform(texts).toarray().astype(np.float32, copy=False)

    X = np.hstack([X_eng.astype(np.float32, copy=False), X_tfidf]).astype(np.float32, copy=False)
    y = np.array(labels, dtype=int)

    spam_count = int(y.sum())
    genuine_count = int(len(y) - spam_count)

    print(f"\nDataset: {len(texts)} samples ({spam_count} spam/fake, {genuine_count} genuine)")
    print(
        f"Feature dimensions: {X_eng.shape[1]} engineered + {X_tfidf.shape[1]} TF-IDF = {X.shape[1]} total"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    gb = HistGradientBoostingClassifier(
        max_iter=80,
        learning_rate=0.08,
        max_depth=6,
        min_samples_leaf=8,
        random_state=42,
    )
    rf = RandomForestClassifier(
        n_estimators=80,
        max_depth=6,
        min_samples_split=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    lr = LogisticRegression(
        C=1.0,
        max_iter=800,
        class_weight='balanced',
        random_state=42,
    )

    ensemble = VotingClassifier(
        estimators=[('gb', gb), ('rf', rf), ('lr', lr)],
        voting='soft',
        weights=[2, 2, 1],
        n_jobs=-1,
    )

    print("\nTraining ensemble model (GB + RF + LR)...")
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    print("\n── Test Set Results ──")
    report = classification_report(y_test, y_pred, target_names=['Genuine', 'Spam/Fake'], digits=4)
    print(report)

    cv_scores = None
    min_class_count = int(np.bincount(y).min()) if len(np.unique(y)) > 1 else 1
    cv_folds = min(3 if len(y) > 20000 else 5, min_class_count)
    if cv_folds >= 2:
        if len(y) > 15000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(y), size=15000, replace=False)
            X_cv = X[idx]
            y_cv = y[idx]
            print(f"Running {cv_folds}-fold CV on a 15,000-sample subset for speed...")
        else:
            X_cv = X
            y_cv = y
        cv_scores = cross_val_score(ensemble, X_cv, y_cv, cv=cv_folds, scoring='f1')
        print(f"{cv_folds}-fold CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble, MODEL_DIR / 'spam_classifier.pkl')
    joblib.dump(tfidf, MODEL_DIR / 'tfidf_vectorizer.pkl')
    joblib.dump(extractor, MODEL_DIR / 'feature_extractor.pkl')

    metadata = {
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'dataset_source': DATASET_SOURCE,
        'dataset_sources': DATASET_SOURCES,
        'dataset_count': len(DATASET_SOURCES),
        'samples': len(texts),
        'spam_samples': spam_count,
        'genuine_samples': genuine_count,
        'feature_count': int(X.shape[1]),
        'cv_f1_mean': round(float(cv_scores.mean()), 4) if cv_scores is not None else None,
        'cv_f1_std': round(float(cv_scores.std()), 4) if cv_scores is not None else None,
    }
    with open(MODEL_DIR / 'model_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print("\n[OK] Saved:")
    print(f"   {MODEL_DIR / 'spam_classifier.pkl'}")
    print(f"   {MODEL_DIR / 'tfidf_vectorizer.pkl'}")
    print(f"   {MODEL_DIR / 'feature_extractor.pkl'}")
    print(f"   {MODEL_DIR / 'model_metadata.json'}")
    print("\nRun: uvicorn app:app --reload --port 8000")


if __name__ == '__main__':
    train(sys.argv[1:] if len(sys.argv) > 1 else None)
