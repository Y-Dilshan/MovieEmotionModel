import argparse
import joblib
import numpy as np

from src.config import MODEL_PATH
from src.data_preprocessing import clean_text


def predict_emotions(plot, top_k=5, threshold=None):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run: python -m src.train"
        )

    artifact = joblib.load(MODEL_PATH)
    model = artifact["pipeline"]
    labels = artifact["labels"]

    if threshold is None:
        threshold = artifact.get("threshold", 0.35)

    cleaned_plot = clean_text(plot)
    if not cleaned_plot:
        raise ValueError("Please provide a non-empty movie plot.")

    probabilities = model.predict_proba([cleaned_plot])[0]
    ranked = sorted(
        zip(labels, probabilities),
        key=lambda item: item[1],
        reverse=True
    )

    # Return labels above threshold; if none qualify, return top predictions.
    selected = [(label, float(score)) for label, score in ranked if score >= threshold]
    if not selected:
        selected = [(label, float(score)) for label, score in ranked[:top_k]]

    return selected[:max(top_k, len(selected))]


def main():
    parser = argparse.ArgumentParser(description="Predict emotions from a movie plot.")
    parser.add_argument("--plot", required=True, help="Movie plot text")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    results = predict_emotions(args.plot, args.top_k, args.threshold)

    print("\nPredicted emotions:")
    for emotion, score in results:
        print(f"{emotion.replace('f2_', '').title():20s} {score * 100:5.1f}%")


if __name__ == "__main__":
    main()
