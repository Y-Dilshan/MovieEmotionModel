
from pathlib import Path

# =====================================
# 1. PROJECT DIRECTORY
# =====================================

# Automatically identifies the main project folder
ROOT_DIR = Path(__file__).resolve().parents[1]


# =====================================
# 2. PROJECT FOLDERS
# =====================================

DATA_DIR = ROOT_DIR / "data"

MODEL_DIR = ROOT_DIR / "models"

REPORT_DIR = ROOT_DIR / "reports"


# =====================================
# 3. DATASET PATHS
# =====================================

MOVIE_DATASET = DATA_DIR / "movie_feelings_dataset.csv"

SYNONYM_DATASET = DATA_DIR / "synonyms_of_feelings.csv"


# =====================================
# 4. TRAINED MODEL PATH
# =====================================

MODEL_PATH = MODEL_DIR / "emotion_model.joblib"


# =====================================
# 5. REPORT PATHS
# =====================================

METRICS_PATH = REPORT_DIR / "evaluation.json"

LABEL_COUNTS_PATH = REPORT_DIR / "label_counts.csv"


# =====================================
# 6. MODEL TRAINING SETTINGS
# =====================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

TOP_K_DEFAULT = 5

# Default emotion prediction threshold
DEFAULT_THRESHOLD = 0.35


# =====================================
# 7. AUTOMATICALLY CREATE FOLDERS
# =====================================

DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================
# 8. DISPLAY PROJECT PATHS
# =====================================

if __name__ == "__main__":

    print("Movie Emotion Prediction Project")

    print("--------------------------------")

    print(f"Project folder : {ROOT_DIR}")

    print(f"Dataset folder : {DATA_DIR}")

    print(f"Model folder   : {MODEL_DIR}")

    print(f"Report folder  : {REPORT_DIR}")

    print(f"Movie dataset  : {MOVIE_DATASET}")

    print(f"Synonyms file  : {SYNONYM_DATASET}")

    print(f"Model file     : {MODEL_PATH}")