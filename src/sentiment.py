"""Reusable components for three-class sentiment classification.

Only the training split is used to fit the TF-IDF vocabulary.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

RANDOM_STATE = 42
_STEMMER = PorterStemmer()
_TOKENS = re.compile(r"[a-z]+(?:'[a-z]+)?")
_URL_EMAIL = re.compile(r"https?://\S+|www\.\S+|\S+@\S+", re.IGNORECASE)
_MENTIONS = re.compile(r"@\w+")


def tokenize_and_stem(text: str) -> list[str]:
    """Tokenize English text, remove common stop words, and stem tokens.

    URLs, emails and @mentions are omitted; hashtag words are retained.
    """
    text = _URL_EMAIL.sub(" ", str(text).lower())
    text = _MENTIONS.sub(" ", text).replace("#", "")
    tokens = _TOKENS.findall(text)
    return [_STEMMER.stem(t) for t in tokens if len(t) > 1 and t not in ENGLISH_STOP_WORDS]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read CSV with text and sentiment columns, removing unusable rows.

    Duplicate *text* rows are removed before splitting to reduce leakage
    between train/validation/test. When duplicate texts have contradictory
    labels, the first occurrence is retained; check those cases separately.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Save your CSV as data/data.csv "
            "or pass --data /path/to/file.csv."
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lower()
    missing = {"text", "sentiment"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Found: {list(df.columns)}")

    df = df[["text", "sentiment"]].dropna().copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df = df[(df["text"] != "") & (df["sentiment"] != "")]
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)

    if df["sentiment"].nunique() < 2:
        raise ValueError("At least two sentiment classes are required for classification.")
    return df


def split_dataset(df: pd.DataFrame, random_state: int = RANDOM_STATE):
    """Stratified split: 70% train, 15% validation, 15% test (approximately)."""
    X = df["text"].to_numpy()
    y = df["sentiment"].to_numpy()
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_and_stem,
        token_pattern=None,  # Custom tokenizer handles tokenization.
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=30_000,
        sublinear_tf=True,
    )
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=2e-4,
        l1_ratio=0.10,
        random_state=random_state,
    )
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def fit_with_validation(
    X_train, y_train, X_val, y_val, *, epochs: int = 12, random_state: int = RANDOM_STATE
):
    """Train SGD classifier; select best epoch using validation *log loss*.

    The test set must not be passed to this function.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    pipeline = make_pipeline(random_state)
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    Xtr = vectorizer.fit_transform(X_train)
    Xva = vectorizer.transform(X_val)
    classes = np.unique(y_train)
    rng = np.random.default_rng(random_state)
    rows = []
    best_loss = float("inf")
    best_classifier = None
    best_epoch = None

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(y_train))
        classifier.partial_fit(Xtr[order], y_train[order], classes=classes)

        p_tr = classifier.predict_proba(Xtr)
        p_va = classifier.predict_proba(Xva)
        train_loss = log_loss(y_train, p_tr, labels=classes)
        val_loss = log_loss(y_val, p_va, labels=classes)
        rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": accuracy_score(y_train, classifier.predict(Xtr)),
            "val_accuracy": accuracy_score(y_val, classifier.predict(Xva)),
        })
        if val_loss < best_loss:
            best_loss = val_loss
            best_classifier = copy.deepcopy(classifier)
            best_epoch = epoch

    # Restore entire fitted estimator state, not just coefficients/intercept.
    pipeline.steps[-1] = ("classifier", best_classifier)
    return pipeline, pd.DataFrame(rows), best_epoch


def evaluate_model(pipeline: Pipeline, X_test, y_test) -> dict:
    """Evaluate *once* on untouched test examples."""
    predictions = pipeline.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "classification_report": classification_report(y_test, predictions, zero_division=0),
    }


def plot_class_balance(y_all, y_train, y_val, y_test):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), sharey=False)
    for ax, values, name in zip(
        axes, [y_all, y_train, y_val, y_test], ["All", "Train", "Validation", "Test"]
    ):
        pd.Series(values).value_counts().sort_index().plot.bar(ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Sentiment")
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Number of texts")
    fig.tight_layout()
    return fig


def plot_training_history(history: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(history["epoch"], history[f"train_{metric}"], marker="o", label="Train")
        ax.plot(history["epoch"], history[f"val_{metric}"], marker="o", label="Validation")
        ax.set_title(f"{metric.title()} by epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.title())
        ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion(pipeline: Pipeline, X_test, y_test):
    labels = pipeline.named_steps["classifier"].classes_
    matrix = confusion_matrix(y_test, pipeline.predict(X_test), labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Test confusion matrix")
    for (r, c), value in np.ndenumerate(matrix):
        ax.text(c, r, f"{value}", va="center", ha="center", color="black")
    fig.tight_layout()
    return fig


def plot_roc(pipeline: Pipeline, X_test, y_test):
    """One-vs-rest ROC for multi-class, ordinary ROC for binary labels."""
    classes = pipeline.named_steps["classifier"].classes_
    proba = pipeline.predict_proba(X_test)
    fig, ax = plt.subplots(figsize=(7, 5))
    if len(classes) == 2:
        positive = (np.asarray(y_test) == classes[1]).astype(int)
        fpr, tpr, _ = roc_curve(positive, proba[:, 1])
        ax.plot(fpr, tpr, label=f"{classes[1]} (AUC={auc(fpr, tpr):.3f})")
    else:
        binary = label_binarize(y_test, classes=classes)
        for i, label in enumerate(classes):
            if len(np.unique(binary[:, i])) < 2:
                continue
            fpr, tpr, _ = roc_curve(binary[:, i], proba[:, i])
            ax.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.set(title="ROC curves (test, one-vs-rest)", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def save_model(pipeline: Pipeline, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path
