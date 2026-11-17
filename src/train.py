import json
import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score
)

from src.config import (
    MODEL_DIR, REPORT_DIR, MODEL_PATH, METRICS_PATH,
    RANDOM_STATE, TEST_SIZE
)
from src.data_preprocessing import load_and_clean_dataset


def main():
    df, label_columns, vocabulary = load_and_clean_dataset()

    X = df["plot_clean"]
    y = df[label_columns].to_numpy(dtype=int)

    # A normal random split is used here. For a larger dataset,
    # iterative multi-label stratification is a useful improvement.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # TF-IDF converts plot text into weighted numerical word features.
    # Word and bigram features capture single words and short phrases.
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
            stop_words=None,  # Keep negation words such as "not" and "never".
            max_features=50000
        )),
        # One binary classifier is trained for each emotion.
        ("classifier", OneVsRestClassifier(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear"
            )
        ))
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)

    metrics = {
        "rows_total": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "number_of_labels": int(len(label_columns)),
        "micro_f1": float(f1_score(y_test, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "micro_precision": float(precision_score(y_test, predictions, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_test, predictions, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_test, predictions)),
        "labels": label_columns
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Store model, label order, threshold and synonym vocabulary together.
    artifact = {
        "pipeline": pipeline,
        "labels": label_columns,
        "threshold": 0.35,
        "synonyms": vocabulary
    }
    joblib.dump(artifact, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # Per-label report helps identify emotions with weak performance.
    report = classification_report(
        y_test, predictions,
        target_names=label_columns,
        output_dict=True,
        zero_division=0
    )
    import pandas as pd
    pd.DataFrame(report).transpose().to_csv(
        REPORT_DIR / "classification_report.csv"
    )

    print("\nTraining completed.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Evaluation saved to: {METRICS_PATH}")
    print("\nMain metrics:")
    for key in ["micro_f1", "macro_f1", "weighted_f1", "micro_precision", "micro_recall", "hamming_loss"]:
        print(f"{key}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
