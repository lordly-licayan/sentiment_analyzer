from google.cloud import storage
from google.api_core.exceptions import NotFound
from urllib.parse import urlparse
from datetime import timedelta

from google.oauth2 import service_account

from managers import BUCKET_NAME, STORAGE_KEY_FILE

credentials = service_account.Credentials.from_service_account_file(STORAGE_KEY_FILE)

storage_client = storage.Client(credentials=credentials)


def get_blob_uri(bucket_name, object_name):
    return f"gs://{bucket_name}/{object_name}"


def get_blob(bucket_name, blob_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob


def file_upload(bucket_name, blob_name, file_stream, mime_type):
    blob = get_blob(bucket_name, blob_name)
    file_stream.seek(0)
    blob.upload_from_file(file_stream, content_type=mime_type)


def file_upload_from_string(
    bucket_name, blob_name, image_content, content_type="text/csv"
):
    blob = get_blob(bucket_name, blob_name)
    blob.upload_from_string(image_content, content_type)


def file_exists(blob_name, bucket_name=BUCKET_NAME):
    blob = get_blob(bucket_name, blob_name)
    try:
        blob.reload()
        return True
    except NotFound:
        return False


def upload(blob_name, file_stream, mime_type, bucket_name=BUCKET_NAME):
    blob_uri = get_blob_uri(bucket_name, blob_name)
    has_already_saved = file_exists(blob_name, bucket_name)

    if not has_already_saved:
        if "csv" in mime_type or "text/csv" in mime_type:
            file_upload_from_string(bucket_name, blob_name, file_stream, mime_type)
        else:
            file_upload(bucket_name, blob_name, file_stream, mime_type)

    return blob_uri


def upload_csv(blob_name, file_stream, mime_type, bucket_name=BUCKET_NAME):
    blob_uri = get_blob_uri(bucket_name, blob_name)
    has_already_saved = file_exists(blob_name, bucket_name)
    if not has_already_saved:
        file_upload(bucket_name, blob_name, file_stream, mime_type)

    return blob_uri


def read_file_from_gcs(blob_name, bucket_name=BUCKET_NAME):
    blob = get_blob(bucket_name, blob_name)
    content = blob.download_as_bytes()
    return content


def delete_blob(blob_name, bucket_name=BUCKET_NAME):
    blob = get_blob(bucket_name, blob_name)
    try:
        blob.delete()
        return True
    except NotFound:
        return False


def parse_gcs_uri(gcs_uri):
    parsed_uri = urlparse(gcs_uri)
    bucket_name = parsed_uri.netloc
    file_name = parsed_uri.path.lstrip("/")

    return bucket_name, file_name


def get_file_bytes(gcs_uri):
    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
    blob = get_blob(bucket_name, blob_name)
    content_bytes = blob.download_as_bytes()

    return content_bytes


def get_blob_link_url(gcs_uri, expiration_time=3600 * 12):
    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
    blob = get_blob(bucket_name, blob_name)

    blob_url = blob.generate_signed_url(
        version="v4", expiration=timedelta(seconds=expiration_time), method="GET"
    )

    return blob_url
