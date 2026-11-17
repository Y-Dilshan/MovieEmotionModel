
import pandas as pd
import numpy as np
import re
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report

# -----------------------------------
# 1. Load datasets
# -----------------------------------

movies = pd.read_csv(
    "datasets/movie_feelings_dataset.csv"
)

synonyms = pd.read_csv(
    "datasets/synonyms_of_feelings.csv"
)

print("Movie dataset:", movies.shape)
print("Synonyms dataset:", synonyms.shape)

# -----------------------------------
# 2. Clean movie data
# -----------------------------------

movies = movies.dropna(
    subset=["plot", "plot_feelings"]
).copy()

movies["plot"] = movies["plot"].astype(str)

# Convert plot feelings into lists
movies["labels"] = movies["plot_feelings"].apply(
    lambda x: [
        emotion.strip().lower()
        for emotion in str(x).split(",")
        if emotion.strip()
    ]
)

# Remove rows without valid labels
movies = movies[
    movies["labels"].apply(len) > 0
].copy()

print("Valid training records:", len(movies))

# -----------------------------------
# 3. Build synonym dictionary
# -----------------------------------

synonym_map = {}

for emotion in synonyms.columns:

    emotion_name = emotion.lower().strip()

    for word in synonyms[emotion].dropna():

        word = str(word).lower().strip()

        if word:
            synonym_map.setdefault(word, set()).add(
                emotion_name
            )

print("Synonym words:", len(synonym_map))

# -----------------------------------
# 4. Synonym-aware text preprocessing
# -----------------------------------

def process_text(text):

    text = str(text).lower()

    words = re.findall(r"\b[a-z]+\b", text)

    detected_emotions = set()

    for word in words:

        if word in synonym_map:

            detected_emotions.update(
                synonym_map[word]
            )

    emotion_features = " ".join(
        "emotion_" + emotion
        for emotion in sorted(detected_emotions)
    )

    return text + " " + emotion_features


movies["processed_plot"] = movies["plot"].apply(
    process_text
)

# -----------------------------------
# 5. Convert labels into multi-label format
# -----------------------------------

mlb = MultiLabelBinarizer()

y = mlb.fit_transform(movies["labels"])

X = movies["processed_plot"]

print("Number of emotion classes:", len(mlb.classes_))

# -----------------------------------
# 6. Split training and testing data
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# -----------------------------------
# 7. Create ML pipeline
# -----------------------------------

model = Pipeline([

    (
        "tfidf",
        TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            sublinear_tf=True
        )
    ),

    (
        "classifier",
        OneVsRestClassifier(
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    )

])

# -----------------------------------
# 8. Train model
# -----------------------------------

print("\nTraining started...")

model.fit(X_train, y_train)

print("Training completed!")

# -----------------------------------
# 9. Evaluate model
# -----------------------------------

probabilities = model.predict_proba(X_test)

# Select top 3 emotions
top3_predictions = np.zeros_like(y_test)

top3_indices = np.argsort(
    probabilities,
    axis=1
)[:, -3:]

for i, indices in enumerate(top3_indices):
    top3_predictions[i, indices] = 1

print("\nMODEL EVALUATION")

print(
    "Micro F1:",
    f1_score(
        y_test,
        top3_predictions,
        average="micro",
        zero_division=0
    )
)

print(
    "Macro F1:",
    f1_score(
        y_test,
        top3_predictions,
        average="macro",
        zero_division=0
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        top3_predictions,
        target_names=mlb.classes_,
        zero_division=0
    )
)

# -----------------------------------
# 10. Save trained model
# -----------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "classes": mlb.classes_,
        "synonym_map": synonym_map
    },
    "models/movie_emotion_model.pkl"
)

print("\nModel saved successfully!")
print("Location: models/movie_emotion_model.pkl")
