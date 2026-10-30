"""Predict sentiment for a user-supplied English sentence."""

import argparse
from pathlib import Path
import joblib


def main():
    parser = argparse.ArgumentParser(description="Predict a text's sentiment")
    parser.add_argument("--text", required=True, help="English sentence to classify")
    parser.add_argument("--model", default="artifacts/sentiment_model.joblib")
    args = parser.parse_args()

    if not Path(args.model).is_file():
        parser.error(f"Model not found: {args.model}. Run train.py first.")
    # Import for the tokenizer class/function reference in serialized pipeline.
    import src.sentiment  # noqa: F401

    model = joblib.load(args.model)  # Only load models from trusted sources.
    predicted_label = model.predict([args.text])[0]
    print("Predicted sentiment:", predicted_label)
    if hasattr(model, "predict_proba"):
        classes = model.named_steps["classifier"].classes_
        probabilities = model.predict_proba([args.text])[0]
        for label, prob in zip(classes, probabilities):
            print(f"  {label}: {prob:.1%}")


if __name__ == "__main__":
    main()
