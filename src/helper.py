import base64
from datetime import datetime
import hashlib
import os
import re
import logging

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from src import (
    DEFAULT_TRAINED_MODEL_NAME,
    EMBEDDER_MODEL,
    JOBS,
    LABEL_MAP,
    SUPPORTED_CLASSIFIERS,
    TEST_DATA_PATH,
    TRAINED_MODEL_DIR,
)
from uuid import uuid4


from src.db.crud.comments import create_comments
from src.db.crud.fileinfo import create_fileinfo
from src.db.crud.trainedmodel import create_trained_model
from src.db.schemas import CommentBase, FileInfoBase, TrainModelForm, TrainedModelBase

# -----------------------------------------------------------
# LOGGING CONFIGURATION
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


def read_data(path):
    """
    Load CSV with columns: 'comment' and 'label'.

    Behavior:
        - Removes commas from comments
        - Raises an error (skips row) when a comment contains a floating number
        - Skips rows with errors (empty comment, empty label, invalid label)
        - Collects error messages with row numbers
        - Returns (comments, labels, errors)
    """
    df = pd.read_csv(path)
    return df


def process_data(df: pd.DataFrame):
    """
    Process DataFrame with columns: 'comment' and 'label'.
    Behavior:
        - Removes commas from comments
        - Raises an error (skips row) when a comment contains a floating number
        - Skips rows with errors (empty comment, empty label, invalid label)
        - Collects error messages with row numbers
        - Returns (comments, labels, errors)
        Args:
        df (pd.DataFrame): DataFrame with 'comment' and 'label' columns
    """

    # --- Validate column presence ---
    if "comment" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'comment' and 'label' columns.")

    # comments = []
    # labels = []
    data = {}
    errors = []

    # --- Process rows one-by-one ---
    for idx, row in df.iterrows():
        row_num = idx + 1

        raw_comment = row["comment"]
        raw_label = row["label"]

        # --- Validate comment ---
        if pd.isna(raw_comment) or str(raw_comment).strip() == "":
            errors.append(f"Row {row_num}: Empty or missing comment.")
            continue

        comment = str(raw_comment).strip()

        if comment.lower() in ["nan", "none"]:
            errors.append(f"Row {row_num}: Comment is 'nan' or 'None'.")
            continue

        # --- Remove commas ---
        comment = comment.replace(",", "")

        # --- Check for floating number (e.g., 1.0, 3.14) ---
        if re.search(r"\b\d+\.\d+\b", comment):
            errors.append(f"Row {row_num}: Comment contains a floating number.")
            continue

        # --- Validate label ---
        if pd.isna(raw_label) or str(raw_label).strip() == "":
            errors.append(f"Row {row_num}: Empty or missing label.")
            continue

        try:
            label = int(raw_label)
        except ValueError:
            errors.append(f"Row {row_num}: Label is not an integer.")
            continue

        if label not in (-1, 0, 1):
            errors.append(f"Row {row_num}: Label must be -1, 0, or 1 (got {label}).")
            continue

        # --- Normalize comment ---
        def normalize(text):
            text = text.lower()
            text = re.sub(r"\s+", " ", text)
            return text

        comment = normalize(comment)

        # --- Store valid row ---
        data[comment] = label
        # comments.append(comment)
        # labels.append(label)

    # If absolutely no valid rows exist, raise an error
    if len(data) == 0:
        raise ValueError(
            "All rows contain errors. No valid data to process.\n" + "\n".join(errors)
        )

    return data, errors


def dict_to_lists(data: dict):
    """
    Docstring for dict_to_lists

    :param data: Description
    :type data: dict
    """
    comments = list(data.keys())
    labels = list(data.values())
    return comments, labels


def convert_label_to_sentiment(labels: list, label_map=LABEL_MAP):
    """
    Convert a list of numeric labels (-1, 0, 1) to sentiment strings.
    label_map must be like {"negative": -1, "neutral": 0, "positive": 1}
    """
    # Invert the map so we can look up by numeric value
    inverse_map = {v: k for k, v in label_map.items()}

    sentiments = []
    for label in labels:
        try:
            sentiments.append(inverse_map[int(label)])
        except (KeyError, ValueError):
            raise ValueError(f"Label {label} is invalid. Allowed labels: -1, 0, 1.")

    return sentiments


def save_comments_to_csv(comments, labels, output_path="output.csv"):
    """
    Save comments and labels into a CSV file with 2 columns:
        - comment
        - label

    Args:
        comments (list[str]): List of text comments
        labels (list[str]): List of labels (same length as comments)
        output_path (str): Path to save the CSV
    """

    if len(comments) != len(labels):
        raise ValueError("The number of comments and labels must be the same.")

    # Create DataFrame
    df = pd.DataFrame({"comment": comments, "label": labels})

    # Save to CSV
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"CSV successfully saved to {output_path}")


def get_embedder(model_name=EMBEDDER_MODEL):
    embedder = SentenceTransformer(model_name)
    return embedder


def generate_school_years():
    current_year = datetime.now().year
    school_years = [f"{y}-{y+1}" for y in range(current_year, 2019, -1)]
    return school_years


def generate_semesters():
    semesters = ["1st", "2nd", "Summer"]
    return semesters


def create_job():
    """
    Create a new job entry in JOBS and return the job_id.
    """

    job_id = str(uuid4())
    JOBS[job_id] = {
        "status": "queued",
        "progress": "",
        "message": "Job created",
        "accuracy": "",
        "elapsedTime": "",
        "errors": "",
        "report": "",
        "feedback": "",
    }
    return job_id


def update_job(job_id, **kwargs):
    """
    Update fields of JOBS[job_id] safely.

    Example:
        update_job(job_id, progress=50, message="Halfway done")
    """
    if job_id not in JOBS:
        raise KeyError(f"Job ID '{job_id}' does not exist in JOBS.")

    for key, value in kwargs.items():
        if key in JOBS[job_id]:
            JOBS[job_id][key] = value
        else:
            # optional: allow dynamic fields OR raise an error
            JOBS[job_id][key] = value  # add new fields if needed


def remove_job(job_id):
    """
    Remove a job entry from JOBS.
    """
    if job_id in JOBS:
        del JOBS[job_id]


def format_seconds(total_seconds):
    """
    Convert seconds to a string in the format 'H hours : M minutes : S seconds'.

    Args:
        total_seconds (int): Number of seconds.

    Returns:
        str: Formatted string.
    """
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    return f"{hours:.0f}:{minutes:.0f}:{seconds:.0f}"


def get_file_hash(content):
    sha256_hash = hashlib.sha256(content).digest()
    file_hash = base64.urlsafe_b64encode(sha256_hash).decode("utf-8").rstrip("=")
    return file_hash


def save_file_info(db, file_id, filename, no_of_data, errors):
    file_info = FileInfoBase(
        file_id=file_id,
        filename=filename,
        no_of_data=no_of_data,
        remarks=str(errors),
    )
    create_fileinfo(db, file_info)


def save_comments(db, file_id, comments, labels, sentiments):
    data = []

    for comment, label, sentiment in zip(comments, labels, sentiments):
        data.append(
            CommentBase(
                file_id=file_id, comment=comment, label=label, remarks=sentiment
            )
        )

    create_comments(db, data)


def save_trained_model(
    db,
    clf,
    data: TrainModelForm,
    accuracy,
    no_of_data,
    remarks,
    model_name=DEFAULT_TRAINED_MODEL_NAME,
    model_dir=TRAINED_MODEL_DIR,
):
    """
    Save the trained model to Google Cloud.
    """
    if os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, model_name)

    joblib.dump(clf, model_path)
    logger.info(f"Model saved to {model_path}")

    trained_model = TrainedModelBase(
        sy=data.sy,
        semester=data.semester,
        model_name=model_name,
        classifier=data.classifierModel,
        accuracy=accuracy,
        no_of_data=no_of_data,
        remarks=remarks,
    )

    create_trained_model(db, trained_model)


def get_trained_model(model_name: str, model_dir=TRAINED_MODEL_DIR):
    """
    Load the trained model from Google Cloud Storage.
    """
    model_path = os.path.join(model_dir, model_name)
    clf = joblib.load(model_path)
    return clf


def get_sentiments(trained_model, payload: str):
    """
    Get sentiments for the given payload using the trained model.
    Args:
        trained_model: Loaded trained model
        payload (dict): Payload containing 'text' key with comments
        Returns:
        dict: Mapping of comment to predicted sentiment label
    """
    text = payload["text"]
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    data = {}
    embedder = get_embedder()
    for line in lines:
        vector = embedder.encode([line], convert_to_numpy=True)
        pred_label = trained_model.predict(vector)[0]
        data[line] = pred_label

    return data
