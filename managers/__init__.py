import os
from dotenv import load_dotenv

load_dotenv()

BUCKET_NAME = os.getenv("BUCKET_NAME", "faculytics-app-storage")
STORAGE_KEY_FILE = os.getenv("STORAGE_KEY_FILE", "faculytics-app-storage.json")
