import re
import unicodedata
import pandas as pd

from src.config import MOVIE_DATASET, SYNONYM_DATASET, REPORT_DIR


def clean_text(text):
    """Normalize a plot without removing useful words such as 'not'."""
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_synonyms(path=SYNONYM_DATASET):
    """
    Convert the wide synonym CSV into:
    {'fear': ['fear', 'afraid', 'alarming', ...], ...}
    Empty cells are ignored.
    """
    synonym_df = pd.read_csv(path)
    vocabulary = {}

    for column in synonym_df.columns:
        emotion = clean_text(column)
        words = []
        for value in synonym_df[column].dropna():
            word = clean_text(value)
            if word and word not in words:
                words.append(word)

        # Always include the emotion name itself.
        if emotion and emotion not in words:
            words.insert(0, emotion)

        if emotion:
            vocabulary[emotion] = words

    return vocabulary


def load_and_clean_dataset(
    movie_path=MOVIE_DATASET,
    synonym_path=SYNONYM_DATASET,
    save_cleaned=True
):
    df = pd.read_csv(movie_path)

    required = {"plot"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Movie dataset is missing required columns: {missing}")

    # Remove rows without usable input text.
    df["plot"] = df["plot"].fillna("").astype(str)
    df["plot_clean"] = df["plot"].apply(clean_text)
    df = df[df["plot_clean"].str.len() > 0].copy()

    # Remove exact duplicate plots. Keep the first occurrence.
    df = df.drop_duplicates(subset=["plot_clean"]).reset_index(drop=True)

    # f2 columns are the primary target labels.
    label_columns = [c for c in df.columns if c.startswith("f2_")]
    if not label_columns:
        raise ValueError("No f2_* target columns were found.")

    # Convert labels safely to 0/1.
    for column in label_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
        df[column] = (df[column] > 0).astype(int)

    # Remove labels that contain only zeros or only ones.
    usable_labels = [
        c for c in label_columns
        if df[c].nunique() == 2 and df[c].sum() > 0
    ]
    removed_labels = sorted(set(label_columns) - set(usable_labels))

    if not usable_labels:
        raise ValueError("No usable f2_* labels remain after cleaning.")

    # Remove records with no positive emotion labels.
    df = df[df[usable_labels].sum(axis=1) > 0].reset_index(drop=True)

    # Load and save the synonym vocabulary as a simple CSV report.
    vocabulary = load_synonyms(synonym_path)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    counts = df[usable_labels].sum().sort_values(ascending=False)
    counts.rename("positive_examples").to_csv(
        REPORT_DIR / "label_counts.csv", header=True
    )

    if save_cleaned:
        clean_columns = ["title", "year", "plot", "plot_clean"] + usable_labels
        clean_columns = [c for c in clean_columns if c in df.columns]
        df[clean_columns].to_csv(
            REPORT_DIR / "cleaned_movie_dataset.csv", index=False
        )

    print(f"Rows after cleaning: {len(df)}")
    print(f"Usable emotion labels: {len(usable_labels)}")
    print(f"Removed constant labels: {removed_labels}")
    print(f"Synonym categories loaded: {len(vocabulary)}")

    return df, usable_labels, vocabulary


if __name__ == "__main__":
    load_and_clean_dataset()
