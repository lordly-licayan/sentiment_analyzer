import os
from dotenv import load_dotenv


load_dotenv()

LABEL_MAP = os.getenv("LABEL_MAP")
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL")
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH")

MODEL_PATH = os.path.join(os.getenv("MODEL_DIR"), os.getenv("MODEL_NAME"))

JOBS = {}  # In-memory job store
