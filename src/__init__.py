import json
import os
from dotenv import load_dotenv

load_dotenv()

LABEL_MAP = json.loads(os.getenv("LABEL_MAP"))
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL")
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH")

TRAINED_MODEL_DIR = os.getenv("TRAINED_MODEL_DIR")
DEFAULT_TRAINED_MODEL_NAME = os.getenv("DEFAULT_TRAINED_MODEL_NAME")

SUPPORTED_CLASSIFIERS = json.loads(os.getenv("SUPPORTED_CLASSIFIERS"))
DEFAULT_CLASSIFIER = os.getenv("DEFAULT_CLASSIFIER")

JOBS = {}  # In-memory job store
