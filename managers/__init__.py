import os
from dotenv import load_dotenv
from google.oauth2 import service_account
from google.cloud import storage

load_dotenv()

BUCKET_NAME = os.getenv("BUCKET_NAME", "faculytics-app")
STORAGE_KEY_FILE = os.getenv("STORAGE_KEY_FILE", "faculytics-app-storage.json")
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "keys.json")


def get_storage_client():
    if os.path.exists(SERVICE_ACCOUNT_FILE):
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE
        )
        client = storage.Client(credentials=creds)
    else:
        client = storage.Client()

    return client
