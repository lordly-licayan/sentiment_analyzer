import os
from dotenv import load_dotenv


load_dotenv()

LABEL_MAP = os.getenv("LABEL_MAP")
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL")
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH")

TRAINED_MODEL_DIR = os.getenv("TRAINED_MODEL_DIR")
DEFAULT_TRAINED_MODEL_NAME = os.getenv("DEFAULT_TRAINED_MODEL_NAME")

SUPPORTED_CLASSIFIERS = os.getenv("SUPPORTED_CLASSIFIERS").split(",")
DEFAULT_CLASSIFIER = os.getenv("DEFAULT_CLASSIFIER")

JOBS = {}  # In-memory job store
