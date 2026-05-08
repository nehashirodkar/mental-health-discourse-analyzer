"""Central config: paths, labels, hyperparameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MLRUNS_DIR = ROOT / "mlruns"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

RAW_CSV = DATA_DIR / "reddit_mh_raw.csv"
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"

CLASSIFIER_DIR = MODELS_DIR / "roberta_distress"
TOPIC_MODEL_DIR = MODELS_DIR / "bertopic"

# Severity taxonomy mapped to mental-health subreddits available in the
# public corpus we use (solomonk/reddit_mental_health_posts: adhd,
# aspergers, depression, ocd, ptsd). We pivoted from a 4-class scheme
# (none/mild/moderate/severe) to 3 classes after observing the corpus
# does not include casual / non-MH subreddits — labeling 'none' would
# require a separate data source.
# 0 = mild, 1 = moderate, 2 = severe
LABELS = ["mild", "moderate", "severe"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

SUBREDDIT_TO_SEVERITY = {
    "adhd": 0,
    "aspergers": 0,
    "depression": 1,
    "ocd": 1,
    "ptsd": 2,
}

# RoBERTa fine-tune
BASE_MODEL = "roberta-base"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

# BERTopic
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_TOPIC_SIZE = 30

# MLflow
MLFLOW_EXPERIMENT = "mental-health-distress"

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
