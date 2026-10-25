"""Basic tests for the prediction module."""

import pytest
from src.predict import predict_message, load_model


def test_model_loads():
    model = load_model()
    assert model is not None


def test_obvious_spam():
    label, score = predict_message(
        "Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121"
    )
    assert label == "spam"


def test_obvious_ham():
    label, score = predict_message("Hey, are we still on for dinner tonight?")
    assert label == "ham"


def test_empty_message_raises():
    with pytest.raises(ValueError):
        predict_message("")


def test_whitespace_only_raises():
    with pytest.raises(ValueError):
        predict_message("   ")


def test_return_types():
    label, score = predict_message("See you tomorrow")
    assert isinstance(label, str)
    assert isinstance(score, float)
    assert label in {"ham", "spam"}
