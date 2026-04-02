import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import joblib
import pandas as pd
from agents.url_agent.feature_extraction import extract_features

# ── Load saved models ──────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
models = joblib.load(os.path.join(BASE_DIR, "model", "url_model_ensemble.pkl"))
feature_cols = joblib.load(os.path.join(BASE_DIR, "model", "feature_columns.pkl"))

def predict_url(url: str) -> dict:
    features = pd.Series(extract_features(url)).to_frame().T
    features = features.reindex(columns=feature_cols, fill_value=0)

    rf_prob = models["rf"].predict_proba(features)[:, 1][0]
    et_prob = models["et"].predict_proba(features)[:, 1][0]
    gb_prob = models["gb"].predict_proba(features)[:, 1][0]
    avg_prob = (rf_prob + et_prob + gb_prob) / 3

    return {
        "url": url,
        "verdict": "🚨 PHISHING" if avg_prob >= 0.5 else "✅ BENIGN",
        "confidence": f"{avg_prob * 100:.2f}%",
        "score": round(avg_prob, 4)
    }

# ── Test URLs ──────────────────────────────
test_urls = [
    # Legit sites
    "https://google.com",
    "https://www.paypal.com",
    "https://facebook.com/login",
    "https://www.amazon.com",
    "https://github.com",

    # Phishing sites
    "http://paypa1.com/secure-login",
    "http://secure-paypal.tk/verify",
    "http://apple.com.id-verify.xyz",
    "https://steamproxy.net/login",
    "http://login-amazon-account.com",
]

print("=" * 55)
print("       URL AGENT — PREDICTION TEST")
print("=" * 55)

for url in test_urls:
    result = predict_url(url)
    print(f"\nURL       : {result['url']}")
    print(f"Verdict   : {result['verdict']}")
    print(f"Confidence: {result['confidence']}")