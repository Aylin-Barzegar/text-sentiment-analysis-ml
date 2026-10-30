import numpy as np
import pandas as pd
import pytest

from src.sentiment import (
    evaluate_model,
    fit_with_validation,
    load_dataset,
    split_dataset,
    tokenize_and_stem,
)


def make_test_data():
    rows = []
    examples = {
        "positive": ["I absolutely love this", "What a great and amazing moment"],
        "negative": ["I really hate this", "What a terrible and awful moment"],
        "neutral": ["The event is scheduled", "The meeting begins today"],
    }
    for label, texts in examples.items():
        for i in range(12):
            rows.append({"text": f"{texts[i % 2]} example {i}", "sentiment": label})
    return pd.DataFrame(rows)


def test_dataset_columns_and_deduplication(tmp_path):
    path = tmp_path / "input.csv"
    pd.DataFrame({
        "id": [1, 2, 3, 4],
        "text": ["Hello", "Hello", "Good news", ""],
        "sentiment": ["POSITIVE", "POSITIVE", "neutral", "negative"],
    }).to_csv(path, index=False)
    loaded = load_dataset(path)
    assert len(loaded) == 2
    assert set(loaded.columns) == {"text", "sentiment"}
    assert set(loaded["sentiment"]) == {"positive", "neutral"}


def test_missing_data_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_dataset(tmp_path / "not_here.csv")


def test_tokenization():
    tokens = tokenize_and_stem("I love #movies! https://example.org @someone")
    assert "love" in tokens
    assert "movi" in tokens
    assert "someon" not in tokens


def test_end_to_end_small_dataset():
    data = make_test_data()
    Xtr, Xval, Xte, ytr, yval, yte = split_dataset(data)
    assert len(Xtr) + len(Xval) + len(Xte) == len(data)
    assert set(np.unique(ytr)) == set(data["sentiment"].unique())
    model, history, best_epoch = fit_with_validation(Xtr, ytr, Xval, yval, epochs=2)
    assert 1 <= best_epoch <= 2
    assert len(history) == 2
    result = evaluate_model(model, Xte, yte)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["macro_f1"] <= 1.0
    assert len(model.predict(["I really love this movie"])) == 1
