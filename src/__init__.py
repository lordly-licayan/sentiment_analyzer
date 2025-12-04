import json
import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDER_MODEL = os.getenv(
    "EMBEDDER_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", "")

LABEL_MAP = json.loads(
    os.getenv("LABEL_MAP", {"negative": -1, "neutral": 0, "positive": 1})
)

SUPPORTED_CLASSIFIERS = json.loads(
    os.getenv(
        "SUPPORTED_CLASSIFIERS", {"LogisticRegression": "lr", "SGDClassifier": "sgd"}
    )
)
DEFAULT_CLASSIFIER = os.getenv("DEFAULT_CLASSIFIER", "LogisticRegression")

LOGISTIC_REGRESSION_MODEL = os.getenv("LOGISTIC_REGRESSION_MODEL", "LogisticRegression")
SGD_CLASSIFIER_MODEL = os.getenv("SGD_CLASSIFIER_MODEL", "SGDClassifier")

TRAINED_MODEL_DIR = os.getenv("TRAINED_MODEL_DIR", "trained_model")
DEFAULT_TRAINED_MODEL_NAME = os.getenv("DEFAULT_TRAINED_MODEL_NAME", "sentiment_model")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
PATIENCE = os.getenv("PATIENCE", 3)

TEACHER_EVALUATION_CATEGORIES = os.getenv("TEACHER_EVALUATION_CATEGORIES", "").split(
    ", "
)

JOBS = {}  # In-memory job store
