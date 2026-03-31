"""
ReviewGuard - ML Model Trainer
Trains a spam detection model using engineered features + labeled data.
Run once: python train_model.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import re
import os
import sys

# ─── TRAINING DATA ──────────────────────────────────────────────────────────
# Realistic labeled review dataset (label: 1=spam, 0=genuine)
TRAINING_DATA = [
    # === GENUINE REVIEWS ===This smartphone has an excellent camera and the battery life is impressive. Highly recommend!
    ("", 0),
    ("The build quality is superb and performance is smooth. Camera captures stunning photos in low light.", 0),
    ("Great product overall. Arrived on time and works exactly as described. Five stars.", 0),
    ("I have been using this for three weeks now and I am very satisfied with the performance.", 0),
    ("Battery lasts all day even with heavy use. The display is vibrant and sharp.", 0),
    ("Solid phone for the price. Camera quality is decent and the software is clean.", 0),
    ("Very happy with my purchase. Setup was easy and it runs all my apps without lag.", 0),
    ("The fingerprint sensor is fast and reliable. Love the design and premium feel.", 0),
    ("Excellent value for money. The processor handles multitasking effortlessly.", 0),
    ("Good phone but the charging speed could be faster. Otherwise it is a great buy.", 0),
    ("I switched from a competitor and this is far better. Display quality is outstanding.", 0),
    ("Screen resolution is crisp and colors are accurate. Very pleased with this purchase.", 0),
    ("Works great for everyday use. The camera has improved significantly from the previous model.", 0),
    ("Solid build, great performance. Call quality is clear and battery charges quickly.", 0),
    ("Impressed with the camera night mode feature. Photos look professional quality.", 0),
    ("The phone heats up a bit during gaming but cools down quickly. Overall good device.", 0),
    ("Interface is intuitive and user friendly. The speaker quality is loud and clear.", 0),
    ("I would recommend this to anyone looking for a reliable mid-range smartphone.", 0),
    ("Great addition to my tech collection. The software updates are timely and smooth.", 0),
    ("Beautiful design and the OLED display makes videos look stunning.", 0),
    ("Slightly expensive but worth every penny for the features you get.", 0),
    ("The zoom feature on the camera is surprisingly good for this price range.", 0),
    ("Fast performance and long battery life make this my go-to daily driver.", 0),
    ("Customer service was helpful when I had a setup question. Product works perfectly.", 0),
    ("Four months in and still running strong without any issues or slowdowns.", 0),
    ("Minimal bloatware and the clean interface makes it easy to use.", 0),
    ("Gaming performance is solid with no frame drops on popular titles.", 0),
    ("The 5G connectivity is blazing fast in my area. Streaming is seamless.", 0),
    ("Charging takes about an hour which is reasonable. Battery capacity is generous.", 0),
    ("Really good value. Storage is ample and the camera handles daylight shots well.", 0),

    # === SPAM / FAKE REVIEWS ===
    ("good", 1),
    ("bad", 1),
    ("ok", 1),
    ("nice", 1),
    ("worst", 1),
    ("best", 1),
    ("not good", 1),
    ("very good", 1),
    ("bad product", 1),
    ("not good not good not good", 1),
    ("BUY NOW AMAZING DEAL CLICK HERE LIMITED OFFER!!!", 1),
    ("BEST PRODUCT EVER BUY BUY BUY!!!!! AMAZING DEAL!!!!", 1),
    ("CLICK HERE FOR FREE GIFT!!! LIMITED TIME OFFER BUY NOW!!!", 1),
    ("WOW WOW WOW AMAZING AMAZING AMAZING BEST BEST BEST!!!", 1),
    ("Visit our website for amazing deals on all products!!!", 1),
    ("Check out mystore.com for 50% off all items!!!!", 1),
    ("Free iPhone giveaway! Click the link in my profile!", 1),
    ("Make $500 daily from home! Visit our site now for details!", 1),
    ("Buy one get one free! Use code SPAM50 at checkout!!!!", 1),
    ("Limited time offer! Don't miss out! Buy now before it's too late!", 1),
    ("good good good good good good good good good good good good", 1),
    ("great great great great great great great great great great", 1),
    ("nice nice nice nice nice nice nice nice nice nice nice nice nice", 1),
    ("amazing amazing amazing amazing amazing amazing amazing amazing", 1),
    ("bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad", 1),
    ("This product is good. This product is good. This product is good.", 1),
    ("Buy this product. Buy this product. Buy this product now.", 1),
    ("!!!!!!!!!!!!!!!!!!!!!!!! AMAZING !!!!!!!!!!!!!!!!!!!!!!!!!!", 1),
    ("#######BEST PRODUCT EVER####### FIVE STARS!!!!!!!!!", 1),
    ("product product product good good good buy buy buy now now now", 1),
    ("I got paid to write this review please buy this amazing product", 1),
    ("Sponsored review: This is a great product, highly recommended!", 1),
    ("aaaaaaaaaaaaaaaaa bbbbbbbbbbbbb cccccccccccc ddddddddd", 1),
    ("asdf qwer zxcv uiop hjkl", 1),
    ("test test test 123 test review test", 1),
    ("xxxxxxxxxxx yyyyyyyyyyy zzzzzzzzz", 1),
    ("1234567890 1234567890 1234567890", 1),
    ("lorem ipsum dolor sit amet consectetur", 1),
    ("temp temp temp placeholder placeholder placeholder", 1),
    ("aaa bbb ccc ddd eee fff ggg hhh iii jjj kkk", 1),
]


# ─── FEATURE ENGINEERING ────────────────────────────────────────────────────
class ReviewFeatureExtractor:
    """Extracts engineered features from review text for ML classification."""
    
    SPAM_PATTERNS = [
        r'buy now', r'click here', r'limited offer', r'free gift',
        r'visit.*website', r'check out.*\.com', r'discount code',
        r'use code', r'sponsored', r'paid.*review', r'make.*\$\d+',
        r'earn money', r'work from home', r'giveaway',
    ]
    
    def __init__(self):
        self.spam_pattern = re.compile('|'.join(self.SPAM_PATTERNS), re.IGNORECASE)
    
    def extract(self, text: str) -> dict:
        if not text:
            text = ""
        
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        
        # Unique word ratio (repetition detection)
        unique_words = set(w.lower() for w in words)
        unique_ratio = len(unique_words) / max(word_count, 1)
        
        # Repetition score: how often the most common word appears
        word_freq = {}
        for w in words:
            word_freq[w.lower()] = word_freq.get(w.lower(), 0) + 1
        max_word_freq = max(word_freq.values(), default=0)
        repetition_score = max_word_freq / max(word_count, 1)
        
        # Caps ratio
        caps_count = sum(1 for c in text if c.isupper())
        caps_ratio = caps_count / max(char_count, 1)
        
        # Punctuation density
        punct_count = sum(1 for c in text if c in '!?.,;:')
        punct_ratio = punct_count / max(char_count, 1)
        
        # Exclamation marks (spam indicator)
        excl_count = text.count('!')
        
        # Special char ratio
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / max(char_count, 1)
        
        # URLs / links
        url_count = len(re.findall(r'https?://|www\.|\.com|\.net|\.org', text, re.IGNORECASE))
        
        # Spam pattern matches
        spam_pattern_hits = len(self.spam_pattern.findall(text))
        
        # Sentence structure (avg sentence length)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        # Digit ratio
        digit_count = sum(1 for c in text if c.isdigit())
        digit_ratio = digit_count / max(char_count, 1)
        
        # Lexical diversity (type-token ratio)
        lex_diversity = len(unique_words) / max(word_count, 1)
        
        # Keyboard smash detection
        consecutive_same = 0
        for i in range(1, len(text)):
            if text[i] == text[i-1]:
                consecutive_same += 1
        consecutive_ratio = consecutive_same / max(char_count, 1)
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'unique_ratio': unique_ratio,
            'repetition_score': repetition_score,
            'caps_ratio': caps_ratio,
            'punct_ratio': punct_ratio,
            'excl_count': min(excl_count, 20),  # cap to avoid outliers
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
def train():
    print("="*60)
    print("ReviewGuard ML Model Trainer")
    print("="*60)
    
    extractor = ReviewFeatureExtractor()
    
    texts = [d[0] for d in TRAINING_DATA]
    labels = [d[1] for d in TRAINING_DATA]
    
    # Engineered features
    X_eng = np.array([extractor.to_array(t) for t in texts])
    
    # TF-IDF features
    tfidf = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 3),
        min_df=1,
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}'
    )
    X_tfidf = tfidf.fit_transform(texts).toarray()
    
    # Combined feature matrix
    X = np.hstack([X_eng, X_tfidf])
    y = np.array(labels)
    
    print(f"\nDataset: {len(texts)} samples ({sum(y)} spam, {len(y)-sum(y)} genuine)")
    print(f"Feature dimensions: {X_eng.shape[1]} engineered + {X_tfidf.shape[1]} TF-IDF = {X.shape[1]} total")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Ensemble: Gradient Boosting + Random Forest + Logistic Regression
    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=4,
        min_samples_split=3, subsample=0.85, random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=6, min_samples_split=3,
        class_weight='balanced', random_state=42
    )
    lr = LogisticRegression(
        C=0.8, max_iter=1000, class_weight='balanced', random_state=42
    )
    
    ensemble = VotingClassifier(
        estimators=[('gb', gb), ('rf', rf), ('lr', lr)],
        voting='soft',
        weights=[3, 2, 1]
    )
    
    print("\nTraining ensemble model (GB + RF + LR)...")
    ensemble.fit(X_train, y_train)
    
    # Evaluation
    y_pred = ensemble.predict(X_test)
    print("\n── Test Set Results ──")
    print(classification_report(y_test, y_pred, target_names=['Genuine','Spam']))
    
    cv_scores = cross_val_score(ensemble, X, y, cv=5, scoring='f1')
    print(f"5-fold CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Save artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(ensemble, 'models/spam_classifier.pkl')
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
    joblib.dump(extractor, 'models/feature_extractor.pkl')
    
    print("\n[OK] Saved:")
    print("   models/spam_classifier.pkl")
    print("   models/tfidf_vectorizer.pkl")
    print("   models/feature_extractor.pkl")
    print("\nRun: uvicorn app:app --reload --port 8000")


if __name__ == '__main__':
    train()
