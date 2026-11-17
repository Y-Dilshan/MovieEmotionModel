
import joblib
import re

# Load trained model
saved_data = joblib.load(
    "models/movie_emotion_model.pkl"
)

model = saved_data["model"]
classes = saved_data["classes"]
synonym_map = saved_data["synonym_map"]


# Same preprocessing used during training
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


# Predict emotions
def predict_emotions(plot):

    processed_plot = process_text(plot)

    probabilities = model.predict_proba(
        [processed_plot]
    )[0]

    top_indices = probabilities.argsort()[-3:][::-1]

    print("\nPredicted Movie Emotions:")

    for index in top_indices:

        emotion = classes[index]
        confidence = probabilities[index] * 100

        print(
            f"{emotion}: {confidence:.2f}%"
        )


# User input
print("MOVIE EMOTION PREDICTION SYSTEM")

movie_plot = input(
    "\nEnter movie description: "
)

predict_emotions(movie_plot)
