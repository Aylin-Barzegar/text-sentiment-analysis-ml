"""Train and evaluate the sentiment classifier from the command line."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save plots without an interactive display.
import matplotlib.pyplot as plt

from src.sentiment import (
    evaluate_model,
    fit_with_validation,
    load_dataset,
    plot_class_balance,
    plot_confusion,
    plot_roc,
    plot_training_history,
    save_model,
    split_dataset,
)


def main():
    parser = argparse.ArgumentParser(description="Train a TF-IDF sentiment classifier")
    parser.add_argument("--data", default="data/data.csv", help="CSV with text and sentiment columns")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--output", default="artifacts/sentiment_model.joblib")
    parser.add_argument("--figures", default="artifacts/figures")
    args = parser.parse_args()

    df = load_dataset(args.data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
    print(f"Dataset: {len(df)} texts; train={len(X_train)} validation={len(X_val)} test={len(X_test)}")
    print("Classes:", sorted(df["sentiment"].unique()))

    figures_dir = Path(args.figures)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_class_balance(df["sentiment"], y_train, y_val, y_test)
    fig.savefig(figures_dir / "class_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    pipeline, history, best_epoch = fit_with_validation(
        X_train, y_train, X_val, y_val, epochs=args.epochs
    )
    print(f"Best epoch (selected on validation log loss): {best_epoch}")

    metrics = evaluate_model(pipeline, X_test, y_test)
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test macro F1: {metrics['macro_f1']:.4f}")
    print(metrics["classification_report"])

    save_model(pipeline, args.output)
    results_dir = Path(args.output).parent
    results_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(results_dir / "training_history.csv", index=False)
    (results_dir / "metrics.json").write_text(
        json.dumps({"best_epoch": best_epoch, **metrics}, indent=2), encoding="utf-8"
    )
    for filename, fig in [
        ("learning_curves.png", plot_training_history(history)),
        ("confusion_matrix.png", plot_confusion(pipeline, X_test, y_test)),
        ("roc_curves.png", plot_roc(pipeline, X_test, y_test)),
    ]:
        fig.savefig(figures_dir / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved model: {args.output}")
    print(f"Saved figures: {figures_dir}")
    print(f"Saved metrics: {results_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
