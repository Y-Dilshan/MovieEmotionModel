from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

MOVIE_DATASET = DATA_DIR / "movie_feelings_dataset.csv"
SYNONYM_DATASET = DATA_DIR / "synonyms_of_feelings.csv"

MODEL_PATH = MODEL_DIR / "emotion_model.joblib"
METRICS_PATH = REPORT_DIR / "evaluation.json"
LABEL_COUNTS_PATH = REPORT_DIR / "label_counts.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TOP_K_DEFAULT = 5
