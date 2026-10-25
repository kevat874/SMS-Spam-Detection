"""
Inference utilities for SMS Spam Detection.
Load the saved pipeline and predict on new messages.
"""

from __future__ import annotations

import joblib
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "spam_classifier.joblib"

_model = None


def load_model():
    """Load the saved pipeline (lazy)."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Please run: python -m src.train"
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_message(text: str) -> Tuple[str, float]:
    """
    Predict whether a message is ham or spam.

    Returns
    -------
    label : str
        'ham' or 'spam'
    confidence : float
        Decision function score (higher = more spam-like for LinearSVC,
        or probability if the model supports it).
    """
    if not text or not text.strip():
        raise ValueError("Message cannot be empty.")

    model = load_model()
    text = text.strip()

    label = model.predict([text])[0]

    # Try to get a meaningful confidence score
    confidence = 0.0
    if hasattr(model.named_steps["clf"], "decision_function"):
        score = model.decision_function([text])[0]
        # For LinearSVC, positive score → spam
        confidence = float(score)
    elif hasattr(model.named_steps["clf"], "predict_proba"):
        proba = model.predict_proba([text])[0]
        # Probability of spam class
        classes = list(model.named_steps["clf"].classes_)
        spam_idx = classes.index("spam")
        confidence = float(proba[spam_idx])

    return label, confidence


if __name__ == "__main__":
    # Quick manual test
    examples = [
        "Congratulations! You have won a free iPhone. Call now 09061701461",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT! Your account has been suspended. Click here to verify.",
        "Can you pick up some milk on your way home?",
    ]
    for msg in examples:
        label, conf = predict_message(msg)
        print(f"[{label.upper():4}] (score={conf:+.3f})  {msg[:60]}...")
