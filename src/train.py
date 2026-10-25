"""
SMS Spam Detection - Complete Training Pipeline
Trains multiple models, evaluates them fairly, selects the best,
and saves a production-ready pipeline.
"""

from __future__ import annotations

import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    make_scorer,
)

# -------------------------------------------------
# Paths & Constants
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "SMSSpamCollection"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TEST_SIZE = 0.20


def load_and_clean() -> pd.DataFrame:
    """Load original UCI data and perform careful cleaning."""
    print("=" * 60)
    print("1. Loading & Cleaning Data")
    print("=" * 60)

    df = pd.read_csv(
        RAW_DATA,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
        quoting=3,
    )
    original_count = len(df)
    print(f"Original rows: {original_count}")

    # Remove exact duplicate rows
    df = df.drop_duplicates()
    after_dup = len(df)
    print(f"Removed {original_count - after_dup} exact duplicates → {after_dup} rows")

    # Strip whitespace and drop empty messages
    df["message"] = df["message"].str.strip()
    df = df[df["message"].str.len() > 0]
    after_empty = len(df)
    print(f"Removed empty messages → {after_empty} rows")

    # Normalize label
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    final_count = len(df)
    print(f"Final cleaned rows: {final_count}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    cleaned_path = PROCESSED_DIR / "sms_cleaned.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"Cleaned data saved → {cleaned_path}")
    return df


def build_pipelines() -> dict:
    """Create Scikit-learn pipelines (vectorizer + classifier)."""
    vectorizer_params = dict(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=5000,
        sublinear_tf=True,
    )

    pipelines = {
        "MultinomialNB": Pipeline([
            ("tfidf", TfidfVectorizer(**vectorizer_params)),
            ("clf", MultinomialNB(alpha=0.5)),
        ]),
        "LogisticRegression": Pipeline([
            ("tfidf", TfidfVectorizer(**vectorizer_params)),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                solver="liblinear",
            )),
        ]),
        "LinearSVC": Pipeline([
            ("tfidf", TfidfVectorizer(**vectorizer_params)),
            ("clf", LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_SEED,
                max_iter=2000,
                dual="auto",
            )),
        ]),
    }
    return pipelines


def evaluate_model(name, pipeline, X_train, y_train, X_test, y_test):
    """Train, time, and evaluate one pipeline."""
    print(f"\n--- {name} ---")
    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = pipeline.predict(X_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="spam"),
        "recall": recall_score(y_test, y_pred, pos_label="spam"),
        "f1": f1_score(y_test, y_pred, pos_label="spam"),
        "train_time_sec": round(train_time, 3),
    }

    print(f"Train time : {metrics['train_time_sec']}s")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}  (spam)")
    print(f"Recall     : {metrics['recall']:.4f}  (spam)")
    print(f"F1-score   : {metrics['f1']:.4f}  (spam)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix (rows=true, cols=pred) [ham, spam]:")
    print(confusion_matrix(y_test, y_pred, labels=["ham", "spam"]))

    return metrics, y_pred


def main():
    print("SMS SPAM DETECTION – FULL TRAINING PIPELINE")
    print("=" * 60)

    df = load_and_clean()

    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Train spam ratio: {(y_train == 'spam').mean():.3f}")
    print(f"Test  spam ratio: {(y_test == 'spam').mean():.3f}")

    pipelines = build_pipelines()
    results = []
    predictions = {}

    for name, pipe in pipelines.items():
        metrics, y_pred = evaluate_model(name, pipe, X_train, y_train, X_test, y_test)
        results.append(metrics)
        predictions[name] = y_pred

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (sorted by F1-spam)")
    print("=" * 60)
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    best_pipeline = pipelines[best_name]
    print(f"\n>>> Selected best model: {best_name}")

    # Cross-validation
    print("\n5-fold Stratified CV on training set (F1-spam):")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    f1_spam = make_scorer(f1_score, pos_label="spam")
    cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=cv, scoring=f1_spam)
    print(f"CV F1 scores: {np.round(cv_scores, 4)}")
    print(f"Mean CV F1  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save
    model_path = MODELS_DIR / "spam_classifier.joblib"
    joblib.dump(best_pipeline, model_path)
    print(f"\nSaved pipeline → {model_path}")

    results_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    print(f"Saved comparison → {MODELS_DIR / 'model_comparison.csv'}")

    # Error analysis
    print("\n" + "=" * 60)
    print("SAMPLE ERROR ANALYSIS (Test set)")
    print("=" * 60)
    y_pred_best = predictions[best_name]
    error_mask = y_test != y_pred_best
    errors = X_test[error_mask]
    true_labels = y_test[error_mask]
    pred_labels = y_pred_best[error_mask]

    print(f"Total errors: {len(errors)}")
    print("\nFalse Negatives (Spam → Ham) – most dangerous:")
    fn_mask = (true_labels == "spam") & (pred_labels == "ham")
    for msg in errors[fn_mask].head(4):
        print(f"  → {msg[:130]}...")

    print("\nFalse Positives (Ham → Spam):")
    fp_mask = (true_labels == "ham") & (pred_labels == "spam")
    for msg in errors[fp_mask].head(4):
        print(f"  → {msg[:130]}...")

    print("\n✅ Training complete. You can now run the Streamlit app.")


if __name__ == "__main__":
    main()
