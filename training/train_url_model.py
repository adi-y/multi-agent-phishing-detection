"""
URL Phishing Detection - Training Script
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
from multiprocessing import Pool, cpu_count, freeze_support
warnings.filterwarnings("ignore")

from agents.url_agent.feature_extraction import extract_features


# ── Worker must be top-level for Windows multiprocessing ─────────────────────
def extract_row(url):
    return extract_features(str(url), live_whois=False, skip_resolve=True)


# ── Everything inside __main__ guard — required on Windows ───────────────────
if __name__ == "__main__":
    freeze_support()

    # ── 1. Load Data ──────────────────────────────────────────────────────
    print("=" * 60)
    print("LOADING DATA...")
    df = pd.read_csv("../datasets/url_data_clean.csv")
    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    # ── 2. Extract Features — parallel ────────────────────────────────────
    n_cores = cpu_count()
    print(f"\nEXTRACTING FEATURES using {n_cores} CPU cores...")

    urls = df["url"].astype(str).tolist()

    with Pool(processes=n_cores) as pool:
        results = pool.map(extract_row, urls)

    # Drop rows where extraction failed
    valid_rows   = [(r, l) for r, l in zip(results, df["label"].tolist()) if r is not None]
    failed       = len(results) - len(valid_rows)
    if failed:
        print(f"[WARNING] {failed} URLs failed feature extraction, dropped.")

    feature_data = pd.DataFrame([r for r, _ in valid_rows])
    labels       = pd.Series([l for _, l in valid_rows])

    print(f"Features extracted: {len(feature_data.columns)}")
    print(f"Samples ready:      {len(feature_data)}")

    # ── 3. Train / Test Split ─────────────────────────────────────────────
    X = feature_data
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # ── 4. Train Three Models ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING MODELS...")

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    et = ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)

    for name, model in [("Random Forest", rf), ("Extra Trees", et), ("Gradient Boosting", gb)]:
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        auc    = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print(f"  → Accuracy: {acc*100:.2f}% | ROC-AUC: {auc:.4f}")

    # ── 5. Ensemble ───────────────────────────────────────────────────────
    print("\nEvaluating Ensemble...")
    avg_proba = (
        rf.predict_proba(X_test)[:, 1] +
        et.predict_proba(X_test)[:, 1] +
        gb.predict_proba(X_test)[:, 1]
    ) / 3
    ens_pred = (avg_proba >= 0.5).astype(int)

    ens_acc = accuracy_score(y_test, ens_pred)
    ens_auc = roc_auc_score(y_test, avg_proba)

    print(f"\n{'=' * 60}")
    print(f"ENSEMBLE RESULTS")
    print(f"{'=' * 60}")
    print(f"Accuracy : {ens_acc*100:.2f}%")
    print(f"ROC-AUC  : {ens_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, ens_pred, target_names=["Benign", "Phishing"]))
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, ens_pred))

    # ── 6. Feature Importance ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("TOP 15 FEATURES (Random Forest):")
    print("=" * 60)
    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(importance.head(15).to_string())

    # ── 7. Save Models ────────────────────────────────────────────────────
    os.makedirs("../model", exist_ok=True)
    joblib.dump({"rf": rf, "et": et, "gb": gb}, "../model/url_model_ensemble.pkl")
    joblib.dump(list(X.columns),                "../model/feature_columns.pkl")

    print(f"\n✅ Models saved: url_model_ensemble.pkl")
    print(f"✅ Feature list saved: feature_columns.pkl")