def predict_emotions(plot, top_k=5, threshold=None):
    artifact = joblib.load(MODEL_PATH)

    model = artifact["pipeline"]
    labels = artifact["labels"]

    cleaned_plot = clean_text(plot)

    probabilities = model.predict_proba(
        [cleaned_plot]
    )[0]

    ranked = sorted(
        zip(labels, probabilities),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked[:top_k]